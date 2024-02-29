# ERA5 to CERRA via Diffusion Models
Wind speed downscaling via Diffusion Models, from ERA5 to CERRA, in the mediterranean region.

### Project Structure

This repository contains several key components that are integral to the project. Below is an overview of each file and its purpose:

- `setup.py`: 
    - **Description**: This file serves as the backbone of the project setup, containing the main parameters and configurations required for initialization. It is essential for setting up the project environment and ensuring all components interact correctly. Refer to the max values contained here for denormalize the results outputted from the model. 

- `utils.py`: 
    - **Description**: A collection of accessory functions that provide support across various modules of the project. Mostly metrics. 

- `generators.py`: 
    - **Description**: Contains the core code for generating sequences dynamically during both the training and evaluation phases. This module is crucial for on-the-fly data processing. It uses memmap to dinamycally load the data, while also performing normalization. 

- `denoising_unet.py`: 
    - **Description**: Implements the denoising U-Net architecture using Keras. 

- `wind_ESPCN-EDSR_downscaling.ipynb`: 
    - **Description**: A Jupyter notebook that includes experiments and analyses related to the ESCN-EDSR downscaling methods. 
- `wind_diffusion_downscaling.ipynb`: 
    - **Description**: This notebook focuses on experiments involving the diffusion model for wind data downscaling.
- `wind_unet_downscaling.ipynb`: 
    - **Description**: Contains experiments using the U-Net model for wind downscaling tasks. 
    - **Warning**: Currently, the weights for the U-Net model are not available in this repository due to memory constraints. The weights are available in the following kaggle [dataset](https://www.kaggle.com/datasets/fastrmerizivic/u-net-weights-for-era5-to-cerra-wind-speed/)
- `graph_visualization_maker.ipynb`: 
    - **Description**: A notebook dedicated to creating graphical representations of the results obtained from various experiments. 
### Reprojection

A guide documenting the reprojection process is available [here](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/how_to_reproject_CERRA.md).

### Data availability

A dataset containing the necessary data in npy format is freely available on [Kaggle](https://www.kaggle.com/datasets/b27f15b82c97022f246b8e525cf75e55b446fc4734af25e767524f9cb62b3f57)

The datasets relative to this project are available from the copernicus website, for [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview), [CERRA](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=form) and [IGRA](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-igra-baseline-network?tab=overview). 

The generated data, missing from CERRA, relative to wind speed in the years 2021-2023 is available at [link](https://www.kaggle.com/datasets/fastrmerizivic/diffusion-generated-cerra-wind-speed-2021-2023). The data is contained in a npz file which include two arrays, the first with wind speed and the second with the relative timestamps. The wind data is given normalized between 0 and 1, for denormalization please refer to the values contained in setup.py. 

### Enviroment 
This project was built using tensorflow 2.15 and keras 2, using a RTX A4000 as a gpu. 

![result](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/bigResult.png)
