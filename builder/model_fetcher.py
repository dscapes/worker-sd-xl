'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import os
import shutil
import requests
import argparse
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

def download_model(model_url: str = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"):
    '''
    Downloads the model from the URL passed in.
    '''
    # read the environment variable
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    # Check if the URL is from huggingface.co, if so, grab the model repo id.
    parsed_url = urlparse(model_url)
    if parsed_url.netloc == "huggingface.co":
        model_id = f"{parsed_url.path.strip('/')}"
    else:
        download_url = model_url
        if parsed_url.netloc == "civitai.com" and args.civitai_token:
            # Добавляем токен к существующим параметрам или создаем новые
            separator = '&' if '?' in model_url else '?'
            download_url = f"{model_url}{separator}token={args.civitai_token}"
            
        downloaded_model = requests.get(download_url, stream=True, timeout=600)
        with open(model_cache_path / "model.zip", "wb") as f:
            for chunk in downloaded_model.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=model_cache_path,
    )

    StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=model_cache_path,
        use_auth_token="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"
    )

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

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
