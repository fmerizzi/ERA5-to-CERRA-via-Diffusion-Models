 ## A quick guide to download and re-project CERRA
ERA5:
- The datasets can be downoaded here: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
- Era5 does not directly provide the wind speed as a parameter, so it must be obtained from the two components u and v with the formula windspeed = sqrt(u^2 + v^2)
- Then you can select years/days/hours (for our experiments we considered 2010-2019 (included) for training, 2020 for testing)
- For Geographical area you should select "sub region extraction" with the followin parameters (explanation of why this coordinates later)
        - North: 47.75
        - South: 35
        - East: 18.75
        - West: 6
- We used GRIB format
- As the time-span we select every 1 every 3 hours to match cerra, starting from 00:00,03:00,06:00 ecc.
- Once the request has been sent you will have to wait before the download becomes available


CERRA:
- The datasets can be downoaded here: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=form
    - CERRA directly provide the windspeed as a parameter
    - level type: surface or atmosphere
    - data type: reanalysis
    - product type: analysis
    - time: select all
    - format: GRIB

Conversion:
- CERRA was built for use troughout europe, including northern europe. For this reason, it was built with a lambert conical projection, to avoid dilatation at higher latitudes. For a direct comparison to take place, we have to reproject CERRA to match ERA-5, which is built with a cylindrical projection.
- To perform the reprojection of CERRA, we used CDO, which is a collection of command line Operators to manipulate and analyse Climate and NWP model Data.
- First of all you need to install it through "sudo apt-get install cdo" or by source: https://www.isimip.org/protocol/preparing-simulation-files/cdo-help/
- Then you need to prepare a configuration file for the coversion. It is a text file named cyl.txt with this content:

    gridtype  = lonlat
    xsize     = 256
    ysize     = 256
    xfirst    = 6
    xinc      = 0.05
    yfirst    = 35
    yinc      = 0.05

Coordinates explanation: please note, here we start from latitue 6 and longitude 35, and we increase with step 0.05 degress for 256 steps, therefore finishing at latitude (6+255*0.05 = 18.75) and longitude (35+255*0.05 = 47.75), which are the coordinates we directly select from the hera form above. We selected this coordinates range to 1) have an image with size in range of our computational means 2) having a map that fully contains italy 3) have an image size divisible by 2 at least a few times. The super-resolution ratio between ERA-5 and CERRA is therefore 52x52 to 256x256, which is around x4.9.

- Then you can convert directly the GRIB file with "cdo remapbil,cyl.txt <inputfile.grib> <outputfile.grib>"


In-situ observations
- The datasets can be downloaded here: https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-igra-baseline-network?tab=form
    - All parameters are similar to those seen for ERA and CERRA.
    - The only difference is that you need to select "CSV one row per observation" in the format section


How to load the data:

Important: if xr.open_dataset doesn't work you may need to install "xarray", "ecmwflibs" and "eccodes" through pip and restart the kernel

    import xarray as xr
    import numpy as np

    # Part valid for both datasets:

    dataset = "dataset path"
    ds_grib = xr.open_dataset(dataset)

    ds_array = ds_grib.to_array()
    ds_numpy = ds_array.to_numpy()

    # Part necessary only for ERA
    # At this point if we print the shape of the ds_numpy_era we should obtain something like
    # (1, X, Y, H, W)
    # And so we need to do a reshape:

    ds_numpy = ds_numpy.reshape(X*Y,H,W)[1:-3]

    # Part necessary only for CERRA

    ds_numpy_cerra = np.squeeze(ds_numpy_cerra)
    ds_numpy_cerra = ds_numpy_cerra[:,::-1,:]      # zero is at the top!

    # We can save the data for further session in .npz format

    np.savez("dstinationPath/destinationName", ds_numpy)

    # Normalization between 0 and 1

    ds_numpy_era = ds_numpy_era / ds_numpy_era.max()
    ds_numpy_cerra = ds_numpy_cerra / ds_numpy_cerra.max()
