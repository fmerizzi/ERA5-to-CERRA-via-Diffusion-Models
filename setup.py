# computed on the whole dataset from 2010-2019 
max_high_res = 31.347172
max_low_res = 26.298004
dataset_lenght_2010_2019 = 29216
dataset_lenght_2020 = 2928
dataset_lenght_2009 = 2920

#num_epochs = 200  # train for at least 50 epochs for good results
image_size = 256
num_frames = 4
plot_diffusion_steps = 20

# sampling

min_signal_rate = 0.015
max_signal_rate = 0.95

# architecture

embedding_dims = 64 # 32
embedding_max_frequency = 1000.0
widths = [64, 128, 256, 384]
block_depth = 3

# optimization

#batch_size =  32
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

