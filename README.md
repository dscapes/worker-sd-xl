# API Reference

## Input Schema

```
{
    "method": "txt2img | txt2img_raw",
    "input": "a photo of a cat"
}
```

txt2img - array of links to the images
txt2img_raw - array of base64 images

```
{
    "method": "get_moldels | add_lora | add_esrgan | add_embedding",
    "input": {
        "url": "https://example.com/my_lora.safetensors"
    }
}
```

```
{
    "method": "get_moldels"
}
```

## Input Schema txt2img | txt2img_raw

```
{
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

## Input Upscale Schema txt2img | txt2img_raw

```
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

