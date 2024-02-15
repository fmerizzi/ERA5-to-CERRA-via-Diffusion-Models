# ERA5 to CERRA via Diffusion Models
Wind speed downscaling via Diffusion Models, from ERA5 to CERRA in the mediterranean region  

### Project Structure

This repository contains several key components that are integral to the project. Below is an overview of each file and its purpose:

- `setup.py`: 
    - **Description**: This file serves as the backbone of the project setup, containing the main parameters and configurations required for initialization. It is essential for setting up the project environment and ensuring all components interact correctly.

- `utils.py`: 
    - **Description**: A collection of accessory functions that provide support across various modules of the project. Mostly metrics. 

- `generators.py`: 
    - **Description**: Contains the core code for generating sequences dynamically during both the training and evaluation phases. This module is crucial for on-the-fly data processing, ensuring efficient and adaptive handling of input data during model training and testing.

- `denoising_unet.py`: 
    - **Description**: Implements the denoising U-Net architecture using Keras. 

- `wind_ESPCN-EDSR_downscaling.ipynb`: 
    - **Description**: A Jupyter notebook that includes experiments and analyses related to the ESCN-EDSR downscaling methods. 
- `wind_diffusion_downscaling.ipynb`: 
    - **Description**: This notebook focuses on experiments involving the diffusion model for wind data downscaling.
- `wind_unet_downscaling.ipynb`: 
    - **Description**: Contains experiments using the U-Net model for wind downscaling tasks. 
    - **Warning**: Currently, the weights for the U-Net model are not available due to memory constraints. This notebook serves as a record of the experimental approach and findings but may require additional resources for full replication.

- `graph_visualization_maker.ipynb`: 
    - **Description**: A notebook dedicated to creating graphical representations of the results obtained from various experiments. 
### Reprojection

A guide documenting the reprojection process is available [here](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/how_to_reproject_CERRA.md).

### Data availability
The generated data, missing from CERRA, relative to wind speed in the years 2021-2023 is available at [link](https://www.kaggle.com/datasets/fastrmerizivic/diffusion-generated-cerra-wind-speed-2021-2023), contained in a npz file which includes two arrays, one with the wind speed and the second with timestamps.

The data relative to this project is available from the copernicus website, for [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview), [CERRA](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=form) and [IGRA](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-igra-baseline-network?tab=overview).  

![result](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/bigResult.png)
