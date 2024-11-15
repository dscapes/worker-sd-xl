''' StableDiffusionXL-v0.9 Predict Module '''

import os
from typing import List
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
#DiffusionPipeline, StableDiffusionXLPipeline #,StableDiffusionXLImg2ImgPipeline,  # для будущего использования
from huggingface_hub._login import _login
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
from PIL import Image

MODEL_CACHE = "diffusers-cache" # todo remove?

token = os.environ.get("HUGGINGFACE_TOKEN", None)
if token is None:
    print('token is None in model_fetcher')

def list_directory_contents(directory):
    return os.listdir(directory)

class Predictor:
    '''Predictor class for StableDiffusionXL-v0.9'''

    def __init__(self, model_tag, hf_token):
        '''
        Initialize the Predictor class
        '''
        self.model_tag = model_tag
        self.device = "cuda"
        self.hf_token = hf_token
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
                
            torch.cuda.empty_cache()
            
            model_path = next(Path("/workspace/models/diffusers-cache").glob("*"))
            model_path = Path(str(model_path).replace('"', '').replace("'", ''))
            
            print(f"Loading model from: {model_path}")
            
            self.base = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.base.to(self.device)
            
            self.refiner = StableDiffusionXLPipeline.from_pretrained(
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
                model_path=f"/workspace/models/esrgan/{model_path}",
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
                num_inference_steps=50, guidance_scale=7.5,
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
        
        # Загружаем и применяем LoRA если они есть
        # if loras:
        #     for lora in loras:
        #         self.base.load_lora_weights(
        #             lora['path'],
        #             weight_name="pytorch_lora_weights.safetensors",
        #             adapter_name=lora['path']
        #         )
        #         self.base.set_adapters(
        #             [lora['path']], 
        #             adapter_weights=[lora['scale']]
        #         )
        # !!! и эмбединги load_embeddings() !!! не забыть про пути

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
            self.refiner.disable_adapters()

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
                    tile_pad=upscale.get('tile_padding', 10),
                    steps=upscale.get('steps', 20)
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
