INPUT_SCHEMA = {
    'method': {
        'type': str,
        'required': False
    },
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'init_image': {
        'type': str,
        'required': False,
        'default': None
    },
    'mask': {
        'type': str,
        'required': False,
        'default': None
    },
    'prompt_strength': {
        'type': float,
        'required': False,
        'default': 0.8,
        'constraints': lambda prompt_strength: 0 <= prompt_strength <= 1
    },
    'num_outputs': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda num_outputs: 10 > num_outputs > 0
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda num_inference_steps: 0 < num_inference_steps < 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 20
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'K-LMS',
        'constraints': lambda scheduler: scheduler in ['DDIM', 'DDPM', 'DPM-M', 'DPM-S',  'EULER-A', 'EULER-D', 'HEUN', 'IPNDM', 'KDPM2-A', 'KDPM2-D', 'PNDM', 'K-LMS', 'KLMS']
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'nsfw': {
        'type': bool,
        'required': False,
        'default': False
    },
    'loras': {
        'type': list,
        'required': False,
        'default': None,
        'schema': {
            'type': dict,
            'schema': {
                'path': {'type': str, 'required': True},
                'scale': {'type': float, 'required': False, 'default': 1.0}
            }
        }
    },
    'upscale': {
        'type': dict,
        'required': False,
        'default': None,
        'schema': {
            'model_path': {'type': str, 'required': True},
            'scale': {'type': float, 'required': False, 'default': 2.0},
            'denoise_strength': {'type': float, 'required': False, 'default': 0.5},
            'steps': {'type': int, 'required': False, 'default': 40},
            'denoising_strength': {'type': float, 'required': False, 'default': 0.8,
                                 'constraints': lambda x: 0.0 <= x <= 1.0},
            'tile_size': {'type': int, 'required': False, 'default': 400},
            'tile_padding': {'type': int, 'required': False, 'default': 10}
        }
    }
}

ADD_LORA_SCHEMA = {
    'method': {
        'type': str,
        'required': False
    },
    'url': {
        'type': str,
        'required': True
    }
}

ADD_ESRGAN_SCHEMA = {
    'method': {
        'type': str,
        'required': False
    },
    'url': {
        'type': str,
        'required': True
    }
}

ADD_EMBEDDING_SCHEMA = {
    'method': {
        'type': str,
        'required': False
    },
    'url': {
        'type': str,
        'required': True
    }
}
