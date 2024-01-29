import setup
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import keras.utils

from setup import *

class DataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, high_res_path, low_res_path, high_max, low_max, num_images, sequential=False, batch_size=24, unet = False):
        # create memory-mapped files for high_res and low_res datasets
        self.high_res = np.memmap(high_res_path, dtype='float32', mode='r', offset = 128, shape=(num_images, 256, 256))
        self.low_res = np.memmap(low_res_path, dtype='float32', mode='r',offset = 128, shape=(num_images, 52, 52))
        # set pre-computed max/min for normalization 
        self.low_max = low_max 
        self.high_max = high_max
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 4
        # flag for diffusion/unet
        self.unet = unet
        self.batch_size = batch_size
        self.num_samples = self.high_res.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches
    
    # must be called to restart the sequential 
    def counter_reset(self):
        self.counter = 0
    
    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.zeros((self.batch_size, 256, 256, self.sequence + 1))
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                items = self.low_res[random + 1:random + self.sequence + 1] / self.low_max
                # swap to channel dimension 
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i, :, :, k] = cv2.resize(items[0, :, :, k], (256, 256), interpolation=cv2.INTER_LINEAR)
                # get the target high res results 
                target = self.high_res[random + self.sequence - 1]
                #normalization
                target = target / self.high_max
                #append the target in the last place 
                inputs[i, :, :, -1] = target
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                items = self.low_res[self.counter + 1:self.counter+self.sequence +1] / self.low_max
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i,:,:,k] = cv2.resize(items[0,:,:,k],(256,256),interpolation=cv2.INTER_LINEAR)
                #get next target (+1) & normalization
                target = self.high_res[self.counter+self.sequence-1] / self.high_max
                inputs[i,:,:,-1] = target
                # update counter value 
                self.counter = self.counter + 1      
        #Diffusion takes a sequence x + y         
        if(self.unet == False):
            return inputs
        # The unet separates intputs and outputs
        elif(self.unet == True):
            return inputs[:,:,:,:-1],inputs[:,:,:,-1]

        
class lowResDataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, high_res_path, low_res_path, high_max, low_max, num_images, sequential=False, batch_size=24, unet = False):
        # create memory-mapped files for high_res and low_res datasets
        self.high_res = np.memmap(high_res_path, dtype='float32', mode='r', offset = 128, shape=(num_images, 256, 256))
        self.low_res = np.memmap(low_res_path, dtype='float32', mode='r',offset = 128, shape=(num_images, 52, 52))
        # set pre-computed max/min for normalization 
        self.low_max = low_max 
        self.high_max = high_max
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 4
        # flag for diffusion/unet
        self.unet = unet
        self.batch_size = batch_size
        self.num_samples = self.high_res.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches
    
    # must be called to restart the sequential 
    def counter_reset(self):
        self.counter = 0
    
    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.zeros((self.batch_size, 52, 52, self.sequence))
        outputs = np.zeros((self.batch_size, 256, 256))
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                items = self.low_res[random + 1:random + self.sequence + 1] / self.low_max
                # swap to channel dimension 
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                inputs[i, :, :, :] = items[0, :, :, :]
                # get the target high res results 
                target = self.high_res[random + self.sequence - 1]
                #normalization
                target = target / self.high_max
                #append the target in the last place 
                outputs[i] = target
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                items = self.low_res[self.counter + 1:self.counter+self.sequence +1] / self.low_max
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                inputs[i, :, :, :] = items[0, :, :, :]
                #get next target (+1) & normalization
                target = self.high_res[self.counter+self.sequence-1] / self.high_max
                outputs[i] = target
                # update counter value 
                self.counter = self.counter + 1      
        return inputs,outputs
 

