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
        self.device = "cuda"
        self.base = None
        self.refiner = None
        self.NSFW = True

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
                
            # Очистим кэш CUDA перед загрузкой модели
            torch.cuda.empty_cache()
            
            self.base = DiffusionPipeline.from_pretrained(
                self.model_tag, #"stabilityai/stable-diffusion-xl-base-1.0"
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.base.to(self.device)
            
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner.to(self.device)

            # Инициализируем словарь для хранения апскейлеров
            self.upsamplers = {}

        except Exception as e:
            print(f"Error setting up the model: {e}")

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

    def load_embeddings(self, embeddings):
        '''
        Load embeddings into the model
        "input": {
            "prompt": "a photo of <cat-toy> in the garden",
            "embeddings": [
                {
                    "path": "cat-toy.pt",
                    "trigger_word": "cat-toy"
                }
            ]
        }
        '''
        for embedding in embeddings:
            self.base.load_textual_inversion(
                embedding['path'],
                token=embedding['trigger_word']
            )
            # Для рефайнера тоже загружаем
            self.refiner.load_textual_inversion(
                embedding['path'],
                token=embedding['trigger_word']
            )

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, seed, 
                num_inference_steps=20, guidance_scale=7.5,
                num_images_per_prompt=1, scheduler='EULER-A',
                loras=None, upscale=None, embeddings=None):
        """
        Генерация изображения
        """
        # Установка seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        # Генерация базового изображения
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="latent",
        ).images

        """
        # Для будущей реализации img2img:
        def img2img(self, prompt, init_image, mask=None, 
                   negative_prompt=None, prompt_strength=0.8,
                   num_inference_steps=50, guidance_scale=7.5,
                   width=512, height=512):
            
            if init_image is not None:
                if mask is not None:
                    # Inpainting
                    image = self.base(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask,
                        negative_prompt=negative_prompt,
                        strength=prompt_strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    ).images
                else:
                    # img2img
                    image = self.base(
                        prompt=prompt,
                        image=init_image,
                        negative_prompt=negative_prompt,
                        strength=prompt_strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    ).images
        """

        output = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=0.8,
            image=image,
        )

        # Отключаем LoRA если использовались
        if loras:
            self.base.disable_adapters()

        # Отключаем эмбединги после использования
        if embeddings:
            self.base.unload_textual_inversion()
            self.refiner.unload_textual_inversion()

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
