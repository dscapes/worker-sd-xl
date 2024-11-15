# RunPod SDXL Serverless API

Docker image builder for [RunPod](https://www.runpod.io/) serverless worker with [API](https://docs.runpod.io/docs/custom-apis).\
Provides a simple API for generating images with SDXL models and their LoRAs.\
Can upscale images with RealESRGAN.\
Downloads Embeddings / LoRAs / ESRGAN models by API.\
Returns a list of Embeddings / LoRAs / ESRGAN models by API.

Default model can be changed by `MODEL_URL` GitHub variable. (Obsolete)

# Common info

Recommended SDXL image resolutions (before upscale):
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

# Serverless API

**TODO:** Add img2img / inpainting / vae / controlnet support.

## Input Schema

```json
{
    "input": {
        "method": "txt2img_raw",
        "prompt": "a photo of a cat",
        "negative_prompt": "bad quality, blurry"
    }
}
```

`txt2img` - array of **links** to the images\
`txt2img_raw` - array of **base64** images

## Input Schema `add_lora` `add_esrgan` `add_embedding`

```json
{
    "input": {
        "method": "add_lora",
        "url": "https://civitai.com/api/download/models/585966"
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

## Input LoRAs Schema `txt2img` `txt2img_raw`

```json
{
    "input": {
        "method": "txt2img_raw",
        "prompt": "1girl, Fenrys",
        "negative_prompt": "bad quality, blurry",
        "width": 1024,
        "height": 1024,
        "seed": 123456,
        "loras": [
            {
                "path": "Fenrys.safetensors",
                "scale": 1.0
            }
        ]
    }
}
```

## Input Upscale Schema `txt2img` `txt2img_raw`

```json
{
    "input": {
        "method": "txt2img_raw",
        "prompt": "a photo of a cat",
        "negative_prompt": "bad quality",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "upscale": {
            "model_path": "4x_foolhardy_Remacri.pth",
            "scale": 2.0,
            "denoise_strength": 0.5,
            "tile_size": 400,
            "tile_padding": 10
        }
    }
}
```

## Input Embeddings Schema `txt2img` `txt2img_raw`

```json
{
    "input": {
        "method": "txt2img_raw",
        "prompt": "a photo of <cat-toy> in the garden",
        "embeddings": [
            {
                "path": "cat-toy.pt",
                "trigger_word": "cat-toy"
            }
        ]
    }
}
```

# Download Models if they don't exist in Network Storage

## RunPod Variables
`CHECKPOINT_URL`
`CHECKPOINT_FILENAME`
`UPSCALER_URL`
`UPSCALER_FILENAME`

## GitHub Variables
`IMAGE_NAME` (Obsolete)
`MODEL_URL` (Obsolete)

## GitHub Secrets
`CIVITAI_TOKEN` (Obsolete)
`DOCKER_HUB_ACCESS_TOKEN`
`DOCKER_HUB_USERNAME`