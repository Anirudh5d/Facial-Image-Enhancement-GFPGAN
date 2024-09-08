import os
import torch
import requests
import cv2
import numpy as np
from basicsr.utils import imwrite
from gfpgan import GFPGANer

# Function to download a file from a URL and check file integrity
def download_file(url, destination, expected_size=None):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded: {destination}")
        
        # Verify file size if expected_size is provided
        if expected_size and os.path.getsize(destination) != expected_size:
            print(f"File size mismatch for {destination}. Expected {expected_size}, got {os.path.getsize(destination)}. Re-downloading...")
            os.remove(destination)  # Remove the corrupted file
            download_file(url, destination, expected_size)
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

# Function to process and restore images using GFPGAN
def enhance_images(input_img, version='1.3', upscale=2, bg_upsampler='realesrgan', weight=0.5, 
                   only_center_face=False, aligned=False, suffix=None, bg_tile=400, ext='auto'):

    # Define the directory to store downloaded models
    model_dir = 'experiments/pretrained_models'
    os.makedirs(model_dir, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            print('The unoptimized RealESRGAN is slow on CPU. Not using it.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = os.path.join(model_dir, 'RealESRGAN_x2plus.pth')
            # Expected size of the RealESRGAN model file (in bytes)
            expected_size = 33039554
            
            # Check if the model is already downloaded and valid
            if not os.path.exists(model_path) or os.path.getsize(model_path) != expected_size:
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                download_file(url, model_path, expected_size)
            
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    arch = 'clean'
    channel_multiplier = 2
    model_name = f'GFPGANv{version}'
    model_path = os.path.join(model_dir, f'{model_name}.pth')
     
    expected_size = 348632874 # this is just to ensure that the model is completely downloaded

    # Check if GFPGAN model is already downloaded and valid
    if not os.path.exists(model_path) or os.path.getsize(model_path) != expected_size:
        url = f'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/{model_name}.pth'
        download_file(url, model_path, expected_size)

    # Initialize GFPGAN restorer
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=aligned,
        only_center_face=only_center_face,
        paste_back=True,
        weight=weight)

    return cropped_faces, restored_faces, restored_img
