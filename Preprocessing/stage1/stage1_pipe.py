from typing import List, Dict, Union, Literal
from sklearn.decomposition import PCA
from numpy import ndarray
import numpy as np
from bm3d import bm3d, BM3DProfile
import cv2
from tqdm import tqdm


def pseduo_four_channel_desnoising(samples):
    denoised_samples: List[Dict] = []
    profile = BM3DProfile()   # create once


    for sample in tqdm(samples, desc="Denoising samples"):
        mosaic = sample["mosaic"].astype(np.float32)  # (H, W)
        H,W = mosaic.shape

        I0_s   = mosaic[0::2, 0::2]  # (H/2, W/2)
        I45_s  = mosaic[0::2, 1::2]
        I90_s  = mosaic[1::2, 0::2]
        I135_s = mosaic[1::2, 1::2]

        pseduo_channels = np.stack([I0_s, I45_s, I90_s, I135_s], axis=0)  # (4, H/2, W/2)
        C, H2, W2 = pseduo_channels.shape

        X = pseduo_channels.reshape(C, -1).T  # (H*W/4, 4)

        pca = PCA(n_components=4, whiten=False)
        X_denoised = pca.fit_transform(X)  # (H*W, 4)
        X_images = X_denoised.T.reshape(C, H2, W2)  # (4, H/2, W/2)
        # Reshaped back to 4 channel

        X_denoised_images = np.empty_like(X_images, dtype=np.float32)
        for c in range(C):
            pc_image = X_images[c].astype(np.float32)
            X_denoised_images[c] = bm3d(
                pc_image,
                sigma_psd=0.005,
                profile=profile,
            )

        X_denoised = X_denoised_images.reshape(C,-1).T  # (H*W, 4)

        X_reconstructed = pca.inverse_transform(X_denoised)  # (H*W, 4)
        raw_denoised = X_reconstructed.T.reshape(C, H2, W2).astype(np.float32)  # (4, H/2, W/2)
        
        I0_d, I45_d, I90_d, I135_d = raw_denoised

        mosaic_denoised = np.zeros(mosaic.shape, dtype=np.float32)
        mosaic_denoised[0::2, 0::2] = I0_d
        mosaic_denoised[0::2, 1::2] = I45_d
        mosaic_denoised[1::2, 0::2] = I90_d
        mosaic_denoised[1::2, 1::2] = I135_d

        
        new_sample = dict(sample)
        new_sample["mosaic"] = mosaic_denoised.astype(np.float32)
        new_sample["meta"]={
             **sample.get("meta", {}),
            "stage1": "pca_denoised",
        }
        denoised_samples.append(new_sample)

    return denoised_samples
    
def interpolate_method(spared_channel, guidance):
    H, W = guidance.shape
    H2, W2 = spared_channel.shape

    upscaled = cv2.resize(spared_channel, (W, H), interpolation=cv2.INTER_LINEAR)

    prediction = upscaled[0::2, 0::2]
    residual = spared_channel - prediction

    residual_upscaled = cv2.resize(residual, (W, H), interpolation=cv2.INTER_LINEAR)

    guided = cv2.ximgproc.guidedFilter(
        guide=guidance.astype(np.float32),
        src=residual_upscaled.astype(np.float32),
        radius=5,
        eps=1e-6,
    )

    reconstruction = upscaled + guided

    return reconstruction.astype(np.float32)


def intensity_guilded_residual_interpolation(denoised_samples):
    sample_images:List[dict] = []
    for sample in tqdm(denoised_samples, desc="IGRI samples"):
        raw_sample = sample["mosaic"]  # ( H, W)

        I0   = raw_sample[0::2, 0::2]
        I45  = raw_sample[0::2, 1::2]
        I90  = raw_sample[1::2, 0::2]
        I135 = raw_sample[1::2, 1::2]

        base_guildance = (I0 + I90   + I45 + I135) / 4.0

        guidance = cv2.resize(base_guildance, (raw_sample.shape[1], raw_sample.shape[0]), interpolation=cv2.INTER_LINEAR)

        I0   = interpolate_method(I0, guidance)
        I45  = interpolate_method(I45, guidance)
        I90  = interpolate_method(I90, guidance)
        I135 = interpolate_method(I135, guidance)

        raw_interpolated = np.stack([I0, I45, I90, I135], axis=0).astype(np.float32)
        new_sample = dict(sample)
        new_sample["mosaic"] = raw_interpolated
        sample_images.append(new_sample)

    # Placeholder for future implementation
    return sample_images