import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import math

from setup import *

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )

    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def channel_attention(x, ratio=16):
    channel_axis = -1
    channel = x.shape[channel_axis]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, activation='linear', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(x)    
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([x, cbam_feature])


def spatial_attention(x):
    avg_out = layers.Lambda(lambda x: keras.backend.mean(x, axis=3, keepdims=True))(x)
    max_out = layers.Lambda(lambda x: keras.backend.max(x, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_out, max_out])
    spatial_attention_feature = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    
    return layers.Multiply()([x, spatial_attention_feature])

def ResidualBlockWithCBAM(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)

        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        # Applying CBAM attention
        x = channel_attention(x)  # Channel attention first
        x = spatial_attention(x)  # Followed by spatial attention

        x = layers.Add()([x, residual])
        return x
    return apply


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        #x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.LayerNormalization(axis=-1,center=True, scale=True)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def attention_block(x, g, width):
    
    # project x and g to width channels
    theta_x = layers.Conv2D(width, [1, 1], strides=[1, 1])(x)
    phi_g = layers.Conv2D(width, [1, 1], strides=[1, 1])(g)
    
    # make the feature map by applying relu on the sum x + g
    f = keras.activations.relu(theta_x + phi_g)
    
    # make the feature with width 1 
    f1 = keras.layers.Conv2D(1, [1, 1], strides=[1, 1])(f)
    # use sigmoid to project into 0-1
    f1 = keras.activations.sigmoid(f1)
    
    #f1 is our attention map, we multiply it by x to obtain the attented values
    attention_applied = keras.layers.Multiply()([x, f1])
    return attention_applied

def ResidualBlockWithAttention(width):
    # More precisely, spatial self attention
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
            
        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        # Apply the attention on x using the residuals as gate 
        x = attention_block(x, residual, width)  # Adjust 'width' based on your preferences for the attention block's inner channels

        x = layers.Add()([x, residual])
        return x
    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            
            #c_attention = channel_attention(x)
            #c_spatial = spatial_attention(x)
            
            #x = layers.Add()([x, c_spatial])
            
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
            
            #c_attention = channel_attention(x)
            #c_spatial = spatial_attention(x)
            
            #x = layers.Add()([x, c_spatial])
        return x

    return apply


def get_network(image_size, input_frames, output_frames, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, input_frames+output_frames))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlockWithAttention(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


def get_post_network(image_size, input_frames, output_frames, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, input_frames))
    #noise_variances = keras.Input(shape=(1, 1, 1))

    #e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    #e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    #x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)
    
    return keras.Model([noisy_images], x, name="residual_unet")
