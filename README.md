# API Reference

## Input Schema

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

## Upscale Schema

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

