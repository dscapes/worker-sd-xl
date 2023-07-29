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

    def __init__(self, model_tag="stabilityai/stable-diffusion-xl-base-0.9"):
        '''
        Initialize the Predictor class
        '''
        self.model_tag = model_tag

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-0.9",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            use_auth_token="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")
        self.txt2img_pipe.to("cuda")
        self.txt2img_pipe.unet = torch.compile(self.txt2img_pipe.unet, mode="reduce-overhead", fullgraph=True)


    @torch.inference_mode()
    def predict(self, prompt, negative_prompt):
        '''
        Run a single prediction on the model
        '''
        output = self.txt2img_pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]

        output_paths = []
        for i, sample in enumerate(output.images):

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        return output_paths
