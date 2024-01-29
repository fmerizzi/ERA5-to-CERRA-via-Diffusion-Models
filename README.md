# ERA5 to CERRA via Diffusion Models

project structure: 

- setup.py : contains the main parameters setting
- utils.py : accessory functions
- generators.py : code that generate sequences on the fly for training and evaluation processes
- denoising_unet.py : keras implementation of the denoising unet used in the diffusion process
- wind_ESPCN-EDSR_downscaling.ipynb : notebook including experiments regarding ESCN-EDSR
- wind_diffusion_downscaling.ipynb : notebook including experiments with diffusion model
- wind_unet_downscaling.ipynb : experiments with U-net (WARNING, weights currently not available because of memory constraints) 


A guide documenting the reprojection process is available [here](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/how_to_reproject_CERRA.md).


Wind speed downscaling via Diffusion Models, from ERA5 to CERRA in the mediterranean region  

![result](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models/blob/main/bigResult.png)
