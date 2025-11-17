This folder is meant to get all images from the camera ready
for the various AI models. It will employ different noise reduction and Demosaicking techniques. 

Citations For proccess

Polarization Denoising and Demosaicking: 
    - https://ieeexplore.ieee.org/document/11084558

Review of polarimetric image denoising:
    -https://researching.cn/articles/OJ730f9435f4156d82 

An Underwater Image Restoration Method With Polarization Imaging Optimization Model for Poor Visible Conditions
    -https://ieeexplore.ieee.org/document/10781421


First step in pipeline. All data will need to be converted
to this format

    {
        "raw":   np.ndarray,  # (4, H, W) in order [I0, I45, I90, I135]
        "label": None,        # placeholder for azimuth labels if you add them later
        "source": "mock",
        "meta":  {...}
    }

This was done so any data when converted to this format should be able to run throught the preprocessing pipeline wihtout issue.