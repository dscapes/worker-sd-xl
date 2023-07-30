''' StableDiffusionXL-v0.9 Predict Module '''

import os
from typing import List

import torch
from diffusers import DiffusionPipeline
from huggingface_hub._login import _login

# from PIL import Image
MODEL_CACHE = "diffusers-cache"

token = os.environ.get("HUGGINGFACE_TOKEN", None)
if token is None:
    print('token is None in model_fetcher')

def list_directory_contents(directory):
    return os.listdir(directory)

class Predictor:
    '''Predictor class for StableDiffusionXL-v0.9'''

    def __init__(self, model_tag="stabilityai/stable-diffusion-xl-base-1.0"):
        '''
        Initialize the Predictor class
        '''
        self.model_tag = model_tag

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            use_auth_token="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")
        self.base.to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")


    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, seed):
        '''
        Run a single prediction on the model
        '''

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        # run both experts
        image = self.base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        output = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        )

        output_paths = []
        for i, sample in enumerate(output.images):

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        return output_paths
