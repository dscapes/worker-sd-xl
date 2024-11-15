'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import os
import shutil
import requests
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from huggingface_hub._login import _login

token = os.environ.get("HUGGINGFACE_TOKEN", None)
if token is None:
    print('token is None in model_fetcher')

SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
MODEL_CACHE_DIR = "diffusers-cache"
ESRGAN_CACHE_DIR = "esrgan"
ESRGAN_URL = "https://civitai.com/api/download/models/164821"

def download_esrgan(url: str = ESRGAN_URL, filename: str = "4x_foolhardy_Remacri.pth"):
    '''
    Downloads the ESRGAN upscaler model
    '''
    esrgan_path = Path(ESRGAN_CACHE_DIR)
    if not esrgan_path.exists():
        esrgan_path.mkdir(parents=True, exist_ok=True)

    downloaded_model = requests.get(url, stream=True, timeout=600)
    
    with open(esrgan_path / filename, "wb") as f:
        for chunk in downloaded_model.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"ESRGAN model downloaded to {esrgan_path / filename}")

def download_model(model_url: str = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"):
    '''
    Downloads the model from the URL passed in.
    '''
    # read the environment variable
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    downloaded_model = requests.get(model_url, stream=True, timeout=600)
    
    # Получаем имя файла из URL или заголовков
    if 'content-disposition' in downloaded_model.headers:
        filename = re.findall("filename=(.+)", downloaded_model.headers['content-disposition'])[0]
    else:
        filename = model_url.split('/')[-1]
        
    # Очищаем имя файла от кавычек
    filename = filename.replace('"', '').replace("'", '')
    
    with open(model_cache_path / filename, "wb") as f:
        for chunk in downloaded_model.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_url", type=str,
    default="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    help="URL of the model to download."
)
parser.add_argument('--civitai_token', type=str, help='Civitai API token')
parser.add_argument('--hf_token', type=str, help='HuggingFace token')

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
    download_esrgan()
