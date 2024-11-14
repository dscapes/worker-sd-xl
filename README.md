# RunPod SDXL Serverless API

[Docker](https://hub.docker.com/) image builder for [RunPod](https://www.runpod.io/) Serverless [API](https://docs.runpod.io/docs/custom-apis).<br>
Provides a simple API for generating images with SDXL models and their LoRAs.<br>
Can upscale images with RealESRGAN and download embeddings, Lora and ESRGAN models to the model list by API.

Default model can be changed by `MODEL_URL` GitHub variable.

# Common info

Recommended XL image resolutions:
```
1024 x 1024 (1:1 Square)
1152 x 896 (9:7)
896 x 1152 (7:9)
1216 x 832 (19:13)
832 x 1216 (13:19)
1344 x 768 (7:4 Horizontal)
768 x 1344 (4:7 Vertical)
1536 x 640 (12:5 Horizontal)
640 x 1536 (5:12 Vertical, the closest to the iPhone resolution)
```

# API Reference

**TODO:** Add `MODEL_URL` override support to the RunPod template. Embedding & inpainting support.

## Input Schema

```json
{
    "input": {
        "method": "txt2img | txt2img_raw",
        "prompt": "a photo of a cat",
        "negative_prompt": "bad quality, blurry"
    }
}
```

`txt2img` - array of **links** to the images

`txt2img_raw` - array of **base64** images

## Input Schema `add_lora` `add_esrgan` `add_embedding`

```json
{
    "input": {
        "method": "add_lora | add_esrgan | add_embedding",
        "url": "https://example.com/my_lora.safetensors"
    }
}
```

## Input Schema `get_moldels` 

```json
{
    "input": {
        "method": "get_models"
    }
}
```

## Input Schema `txt2img` `txt2img_raw`

```json
{
    "method": "txt2img | txt2img_raw",
    "prompt": "a photo of a cat",
    "negative_prompt": "bad quality, blurry",
    "width": 512,
    "height": 512,
    "seed": 123456,
    "loras": [
        {
            "path": "my_lora.safetensors",
            "scale": 0.75
        },
        {
            "path": "another_lora.safetensors",
            "scale": 0.5
        }
    ]
}
```

## Input Upscale Schema `txt2img` `txt2img_raw`

```json
{
    "prompt": "a photo of a cat",
    "negative_prompt": "bad quality",
    "width": 512,
    "height": 512,
    "steps": 40,
    "denoising_strength": 0.8,
    "loras": [
        {
            "path": "my_lora.safetensors",
            "scale": 0.75
        }
    ],
    "upscale": {
        "model_path": "RealESRGAN_x4plus.pth",
        "scale": 2.0,
        "denoise_strength": 0.5,
        "tile_size": 400,
        "tile_padding": 10
    }
}
```

