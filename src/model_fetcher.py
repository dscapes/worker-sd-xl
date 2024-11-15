'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import os
import requests
from pathlib import Path

MODEL_CACHE_DIR = "/workspace/models/diffusers-cache"
ESRGAN_CACHE_DIR = "/workspace/models/esrgan"

def download_if_needed():
    '''
    Downloads models if they don't exist
    '''
    checkpoint_url = os.environ.get('CHECKPOINT_URL')
    checkpoint_filename = os.environ.get('CHECKPOINT_FILENAME')
    upscaler_url = os.environ.get('UPSCALER_URL')
    upscaler_filename = os.environ.get('UPSCALER_FILENAME')

    if checkpoint_filename and not (Path(MODEL_CACHE_DIR) / checkpoint_filename).exists():
        if checkpoint_url:
            print(f"Downloading checkpoint model to {checkpoint_filename}")
            download_model(checkpoint_url, checkpoint_filename)
        else:
            print("CHECKPOINT_URL not provided")

    if upscaler_filename and not (Path(ESRGAN_CACHE_DIR) / upscaler_filename).exists():
        if upscaler_url:
            print(f"Downloading upscaler model to {upscaler_filename}")
            download_esrgan(upscaler_url, upscaler_filename)
        else:
            print("UPSCALER_URL not provided")

def download_model(url: str, filename: str):
    model_cache_path = Path(MODEL_CACHE_DIR)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_model = requests.get(url, stream=True, timeout=600)
    with open(model_cache_path / filename, "wb") as f:
        for chunk in downloaded_model.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def download_esrgan(url: str, filename: str):
    esrgan_path = Path(ESRGAN_CACHE_DIR)
    esrgan_path.mkdir(parents=True, exist_ok=True)

    downloaded_model = requests.get(url, stream=True, timeout=600)
    with open(esrgan_path / filename, "wb") as f:
        for chunk in downloaded_model.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    download_if_needed()
