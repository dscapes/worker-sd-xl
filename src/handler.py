''' infer.py for runpod worker '''

import os
import predict
import argparse
import base64

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup

from schema import INPUT_SCHEMA, ADD_LORA_SCHEMA, ADD_ESRGAN_SCHEMA, ADD_EMBEDDING_SCHEMA

def download_and_save(url, save_dir):
    '''
    Download a file from a URL and save it to the specified directory.
    '''
    file_path = rp_download.download_file_from_url(url)
    if file_path:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        os.rename(file_path, save_path)
        return save_path
    return None

def get_models():
    '''
    Get a list of all LoRA, ESRGAN, and embedding models on the server.
    '''
    lora_models = os.listdir('/loras') if os.path.exists('/loras') else []
    esrgan_models = os.listdir('/esrgan') if os.path.exists('/esrgan') else []
    embedding_models = os.listdir('/embeddings') if os.path.exists('/embeddings') else []

    return {
        "loras": lora_models,
        "esrgan": esrgan_models,
        "embeddings": embedding_models
    }

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_method = job['method']
    job_input = job['input']

    # Input validation
    if job_method == "txt2img":
        validated_input = validate(job_input, INPUT_SCHEMA)
    elif job_method == "add_lora":
        validated_input = validate(job_input, ADD_LORA_SCHEMA)
    elif job_method == "add_esrgan":
        validated_input = validate(job_input, ADD_ESRGAN_SCHEMA)
    elif job_method == "add_embedding":
        validated_input = validate(job_input, ADD_EMBEDDING_SCHEMA)
    else:
        validated_input = {}

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input.get('validated_input', job_input)

    # Download input objects
    validated_input['init_image'], validated_input['mask'] = rp_download.download_files_from_urls(
        job['id'],
        [validated_input.get('init_image', None), validated_input.get(
            'mask', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking

    MODEL.NSFW = validated_input.get('nsfw', True)

    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

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
    elif job_method == "get_models":
        return get_models()
    else:
        return {"error": "Invalid method"}

def handle_txt2img(validated_input, job):
    '''
    Handle txt2img method.
    '''
    loras = validated_input.get("loras", None)
    if loras is not None:
        # Преобразуем пути для каждой LoRA
        for lora in loras:
            lora['path'] = f"/loras/{lora['path']}"
            if 'scale' not in lora:
                lora['scale'] = 1.0  # значение по умолчанию

    img_paths = MODEL.predict(
        prompt=validated_input["prompt"],
        negative_prompt=validated_input.get("negative_prompt", None),
        width=validated_input.get('width', 512),
        height=validated_input.get('height', 512),
        seed=validated_input['seed'],
        loras=loras  # Добавляем параметр loras
    )

    job_output = []
    for index, img_path in enumerate(img_paths):
        file_name = f"{job['id']}_{index}.png"
        image_url = upload_file_to_bucket(file_name, img_path)

        job_output.append({
            "image": image_url,
            "seed": validated_input['seed'] + index
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
        # Преобразуем пути для каждой LoRA
        for lora in loras:
            lora['path'] = f"/loras/{lora['path']}"
            if 'scale' not in lora:
                lora['scale'] = 1.0  # значение по умолчанию

    img_paths = MODEL.predict(
        prompt=validated_input["prompt"],
        negative_prompt=validated_input.get("negative_prompt", None),
        width=validated_input.get('width', 512),
        height=validated_input.get('height', 512),
        seed=validated_input['seed'],
        loras=loras  # Добавляем параметр loras
    )

    job_output = []
    for index, img_path in enumerate(img_paths):
        # Читаем изображение в base64
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        job_output.append({
            "image": f"data:image/png;base64,{encoded_string}",
            "seed": validated_input['seed'] + index
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
parser.add_argument('--model_tag', type=str, default="stabilityai/stable-diffusion-xl-base-0.9")

if __name__ == "__main__":
    args = parser.parse_args()

    MODEL = predict.Predictor(model_tag=args.model_tag)
    MODEL.setup()

    runpod.serverless.start({"handler": run})
