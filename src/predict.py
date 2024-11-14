''' StableDiffusionXL-v0.9 Predict Module '''

import os
from typing import List

import torch
from diffusers import DiffusionPipeline
from huggingface_hub._login import _login
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
from PIL import Image

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

        # Инициализируем словарь для хранения апскейлеров
        self.upsamplers = {}

    def get_upsampler(self, model_path, scale=2, tile=400, tile_pad=10):
        '''
        Создает или возвращает существующий апскейлер для конкретной модели
        '''
        if model_path not in self.upsamplers:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            self.upsamplers[model_path] = RealESRGANer(
                scale=scale,
                model_path=f"/esrgan/{model_path}",
                model=model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=0,
                half=True
            )
        return self.upsamplers[model_path]

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, seed, 
                steps=40, denoising_strength=0.8,
                loras=None, upscale=None):
        '''
        Run a single prediction on the model
        '''
        # Загружаем и применяем LoRA если они есть
        if loras:
            for lora in loras:
                self.base.load_lora_weights(
                    lora['path'],
                    weight_name="pytorch_lora_weights.safetensors",
                    adapter_name=lora['path']
                )
                self.base.set_adapters(
                    [lora['path']], 
                    adapter_weights=[lora['scale']]
                )

        # Генерация базового изображения
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            denoising_end=denoising_strength,
            output_type="latent",
        ).images

        output = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            denoising_start=denoising_strength,
            image=image,
        )

        # Отключаем LoRA если использовались
        if loras:
            self.base.disable_adapters()

        output_paths = []
        for i, sample in enumerate(output.images):
            if upscale:
                # Получаем апскейлер для конкретной модели
                upsampler = self.get_upsampler(
                    upscale['model_path'],
                    scale=upscale.get('scale', 2.0),
                    tile=upscale.get('tile_size', 400),
                    tile_pad=upscale.get('tile_padding', 10)
                )
                
                # Конвертируем PIL Image в numpy array
                img_np = np.array(sample)
                # Upscale изображения
                upscaled, _ = upsampler.enhance(
                    img_np, 
                    outscale=upscale.get('scale', 2.0),
                    denoise_strength=upscale.get('denoise_strength', 0.5)
                )
                # Конвертируем обратно в PIL
                sample = Image.fromarray(upscaled)

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        return output_paths
