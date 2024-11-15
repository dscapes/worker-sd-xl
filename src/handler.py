''' infer.py for runpod worker '''

import os
import predict
import argparse
import base64
import requests
from urllib.parse import urlparse

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup

from schema import INPUT_SCHEMA, ADD_LORA_SCHEMA, ADD_ESRGAN_SCHEMA, ADD_EMBEDDING_SCHEMA

def download_and_save(url, save_dir):
    '''
    Download a file from a URL and save it to the specified directory.
    '''
    try:
        parsed_url = urlparse(url)
        download_url = url
        
        if parsed_url.netloc == "civitai.com" and args.civitai_token:
            separator = '&' if '?' in url else '?'
            download_url = f"{url}{separator}token={args.civitai_token}"
            
        response = requests.get(download_url)
        response.raise_for_status()
        
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition and 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"\'')
        else:
            path_parts = parsed_url.path.rstrip('/').split('/')
            model_id = path_parts[-1]
            
            if save_dir == '/esrgan':
                ext = '.pth'
            elif save_dir == '/embeddings':
                ext = '.pt'
            else:
                ext = '.safetensors'
            filename = f"{model_id}{ext}"
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        return save_path
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None

def get_models():
    '''
    Get a list of all LoRA, ESRGAN, and embedding models on the server.
    '''
    base_models = os.listdir('/workspace/models/diffusers-cache') if os.path.exists('/workspace/models/diffusers-cache') else []
    lora_models = os.listdir('/workspace/models/lora') if os.path.exists('/workspace/models/lora') else []
    esrgan_models = os.listdir('/workspace/models/esrgan') if os.path.exists('/workspace/models/esrgan') else []
    embedding_models = os.listdir('/workspace/models/embeddings') if os.path.exists('/workspace/models/embeddings') else []

    return {
        "checkpoints": base_models,
        "loras": lora_models,
        "esrgan": esrgan_models,
        "embeddings": embedding_models
    }

def run(job):
    '''
    Run inference on the model.
    '''
    print("Received job:", job)

    job_input = job['input']
    if 'method' not in job_input:
        return {"error": "Method is required in input"}

    job_method = job_input['method']

    # Input validation
    if job_method == "txt2img":
        validated_input = validate(job_input, INPUT_SCHEMA)
    elif job_method == "txt2img_raw":
        validated_input = validate(job_input, INPUT_SCHEMA)
    elif job_method == "add_lora":
        validated_input = validate(job_input, ADD_LORA_SCHEMA)
    elif job_method == "add_esrgan":
        validated_input = validate(job_input, ADD_ESRGAN_SCHEMA)
    elif job_method == "add_embedding":
        validated_input = validate(job_input, ADD_EMBEDDING_SCHEMA)
    elif job_method == "get_models":
        return get_models()
    else:
        return {"error": "Invalid method"}

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input.get('validated_input', job_input)

    # Download input objects
    validated_input['init_image'], validated_input['mask'] = rp_download.download_files_from_urls(
        job['id'],
        [validated_input.get('init_image', None), validated_input.get('mask', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking

    MODEL.NSFW = validated_input.get('nsfw', True)

    # Router for different job methods
    if job_method == "txt2img":
        return handle_txt2img(validated_input, job)
    elif job_method == "txt2img_raw":
        return handle_txt2img_raw(validated_input, job)
    elif job_method == "add_lora":
        return handle_add_lora(validated_input)
    elif job_method == "add_esrgan":
        return handle_add_esrgan(validated_input)
    elif job_method == "add_embedding":
        return handle_add_embedding(validated_input)
    else:
        return {"error": "Invalid method"}

def handle_txt2img(validated_input, job):
    '''
    Handle txt2img method.
    '''
    loras = validated_input.get("loras", None)
    if loras is not None:
        for lora in loras:
            lora['path'] = f"/workspace/models/lora/{lora['path']}"
            if 'scale' not in lora:
                lora['scale'] = 1.0

    embeddings = validated_input.get("embeddings", None)
    if embeddings is not None:
        for embedding in embeddings:
            embedding['path'] = f"/workspace/models/embeddings/{embedding['path']}"

    seed = validated_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    img_paths = MODEL.predict(
        prompt=validated_input["prompt"],
        negative_prompt=validated_input.get("negative_prompt"),
        width=validated_input.get("width", 512),
        height=validated_input.get("height", 512),
        seed=seed,
        num_inference_steps=validated_input.get("num_inference_steps", 20),
        guidance_scale=validated_input.get("guidance_scale", 7.5),
        num_images_per_prompt=validated_input.get("num_outputs", 1),
        scheduler=validated_input.get("scheduler", "EULER-A"),
        loras=loras,
        embeddings=embeddings,
        upscale=validated_input.get("upscale")
    )

    job_output = []
    for index, img_path in enumerate(img_paths):
        file_name = f"{job['id']}_{index}.png"
        image_url = upload_file_to_bucket(file_name, img_path)

        job_output.append({
            "image": image_url,
            "seed": seed + index
        })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output

def handle_txt2img_raw(validated_input, job):
    '''
    Handle txt2img method.
    '''
    loras = validated_input.get("loras", None)
    if loras is not None:
        for lora in loras:
            lora['path'] = f"/workspace/models/lora/{lora['path']}"
            if 'scale' not in lora:
                lora['scale'] = 1.0 

    embeddings = validated_input.get("embeddings", None)
    if embeddings is not None:
        for embedding in embeddings:
            embedding['path'] = f"/workspace/models/embeddings/{embedding['path']}"

    seed = validated_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    img_paths = MODEL.predict(
        prompt=validated_input["prompt"],
        negative_prompt=validated_input.get("negative_prompt", None),
        width=validated_input.get('width', 512),
        height=validated_input.get('height', 512),
        seed=seed,
        loras=loras,
        upscale=validated_input.get('upscale', None)
    )

    job_output = []
    for index, img_path in enumerate(img_paths):
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        job_output.append({
            "image": f"data:image/png;base64,{encoded_string}",
            "seed": seed + index
        })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output

def handle_add_lora(validated_input):
    '''
    Handle adding a LoRA model.
    '''
    url = validated_input.get("url")
    if url:
        save_path = download_and_save(url, "/loras")
        if save_path:
            return {"status": "LoRA model added", "path": save_path}
    return {"error": "Failed to add LoRA model"}

def handle_add_esrgan(validated_input):
    '''
    Handle adding an ESRGAN model.
    '''
    url = validated_input.get("url")
    if url:
        save_path = download_and_save(url, "/esrgan")
        if save_path:
            return {"status": "ESRGAN model added", "path": save_path}
    return {"error": "Failed to add ESRGAN model"}

def handle_add_embedding(validated_input):
    '''
    Handle adding an embedding.
    '''
    url = validated_input.get("url")
    if url:
        save_path = download_and_save(url, "/embeddings")
        if save_path:
            return {"status": "Embedding added", "path": save_path}
    return {"error": "Failed to add embedding"}

# Grab args
parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument('--civitai_token', type=str, help='Civitai API token')
parser.add_argument('--hf_token', type=str, help='HuggingFace token')

if __name__ == "__main__":
    args = parser.parse_args()

    # Проверяем и скачиваем модели если нужно
    from model_fetcher import download_if_needed
    download_if_needed()

    MODEL = predict.Predictor(model_tag=args.model_tag, hf_token=args.hf_token)
    MODEL.setup()

    runpod.serverless.start({"handler": run})
