import runpod
import torch
from PIL import Image
from io import BytesIO
import base64
import requests
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
    get_peft_kwargs,
    is_peft_version,
    get_adapter_name,
)
import os
# from huggingface_hub import login

# # Retrieve the token from the environment variable
# hf_token = os.getenv("HUGGINGFACE_TOKEN")

# # Login using the token (no need for manual login)
# login(token=hf_token)

# Padding logic to make the image square
def add_padding_to_square(image, target_size=1024, pad_color='black'):
    width, height = image.size
    if width == height:
        return image.resize((target_size, target_size), Image.LANCZOS), (0, 0, target_size, target_size)
    
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    new_image = Image.new('RGB', (target_size, target_size), color=pad_color)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image, (paste_x, paste_y, new_width, new_height)

# Logic to crop the image back to its original aspect ratio
def crop_to_original_ratio(image, original_size, pad_info):
    orig_width, orig_height = original_size
    paste_x, paste_y, new_width, new_height = pad_info
    
    crop_left = paste_x
    crop_top = paste_y
    crop_right = paste_x + new_width
    crop_bottom = paste_y + new_height
    
    cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cropped_image.resize((orig_width, orig_height), Image.LANCZOS)

# Custom method to load LoRA weights into the transformer
def load_lora_into_transformer(
    cls, state_dict, network_alphas, transformer, adapter_name=None, _pipeline=None
):
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer_keys = [k for k in state_dict if k.startswith(cls.transformer_name)]
    state_dict = {
        k.replace(f"{cls.transformer_name}.", ""): v
        for k, v in state_dict.items()
        if k in transformer_keys
    }

    if state_dict and "lora_A" not in next(iter(state_dict.keys())):
        state_dict = convert_unet_state_dict_to_peft(state_dict)

    if adapter_name in getattr(transformer, "peft_config", {}):
        raise ValueError(f"Adapter name {adapter_name} already exists.")

    rank = {key: val.shape[1] for key, val in state_dict.items() if "lora_B" in key}

    if network_alphas:
        prefix = cls.transformer_name
        network_alphas = {
            k.replace(f"{prefix}.", ""): v
            for k, v in network_alphas.items()
            if k.startswith(prefix)
        }

    lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)
    if "use_dora" in lora_config_kwargs and is_peft_version("<", "0.9.0"):
        raise ValueError("Upgrade peft to 0.9.0+ to use DoRA-enabled LoRAs.")
    lora_config_kwargs.pop("use_dora", None)

    lora_config = LoraConfig(**lora_config_kwargs)
    adapter_name = adapter_name or get_adapter_name(transformer)

    inject_adapter_in_model(
        lora_config, transformer, adapter_name=adapter_name, low_cpu_mem_usage=True
    )
    incompatible_keys = set_peft_model_state_dict(
        transformer, state_dict, adapter_name, low_cpu_mem_usage=True
    )

    if incompatible_keys and hasattr(incompatible_keys, "unexpected_keys"):
        print(f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")

FluxControlNetPipeline.load_lora_into_transformer = classmethod(load_lora_into_transformer)

# Load models into memory
base_model_path = "/workspace/FLUX.1-dev"
controlnet_model_path = "/workspace/FLUX.1-dev-Controlnet-Union"
lora_safetensors_path = "/workspace/lora.safetensors"


# Load ControlNet model from the local path
controlnet = FluxControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.bfloat16)

# Load the base pipeline with the local ControlNet model
pipe = FluxControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipe.load_lora_weights(lora_safetensors_path)

# Move the pipeline to GPU
pipe.to("cuda")

print("Models loaded successfully from local directories.")


# Default prompt with a placeholder
default_prompt = (
    "bybasa style cartoonized image of <<your_text_here>>, "
    "bybasa vector illustration, this image is in the style of bybasa"
)

# Function to replace the placeholder with user-provided text
def replace_placeholder(prompt, user_text):
    return prompt.replace("<<your_text_here>>", user_text)

# Input schema validation function
def validate_inputs(inputs):
    required_fields = ["user_text", "image_data"]
    for field in required_fields:
        if field not in inputs:
            raise ValueError(f"Missing required input: {field}")

    if not isinstance(inputs.get("controlnet_conditioning_scale", 0.31), float):
        raise ValueError("controlnet_conditioning_scale must be a float.")

    if not isinstance(inputs.get("num_generations", 4), int):
        raise ValueError("num_generations must be an integer.")

    if not isinstance(inputs.get("num_inference_steps", 24), int):
        raise ValueError("num_inference_steps must be an integer.")

    if not isinstance(inputs.get("guidance_scale", 5.5), float):
        raise ValueError("guidance_scale must be a float.")

# Function to convert a file to Base64
def file_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Handler function to process the job
def handler(job):
    inputs = job.get("input", {})

    # Validate inputs
    try:
        validate_inputs(inputs)
    except ValueError as e:
        return {"error": str(e)}

    # Extract inputs with defaults where applicable
    prompt = inputs.get("prompt", default_prompt)
    user_text = inputs.get("user_text", "an art by bybasa")
    prompt = replace_placeholder(prompt, user_text)

    controlnet_conditioning_scale = inputs.get("controlnet_conditioning_scale", 0.31)
    control_mode = inputs.get("control_mode", 3)
    lora_scale = inputs.get("lora_scale", 1.0)
    num_generations = inputs.get("num_generations", 4)
    num_inference_steps = inputs.get("num_inference_steps", 24)
    guidance_scale = inputs.get("guidance_scale", 5.5)

    # Handle user-provided image (URL or Base64)
    image_data = inputs["image_data"]
    if image_data.startswith("http"):
        control_image = Image.open(requests.get(image_data, stream=True).raw).convert("RGB")
    else:
        control_image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

    # Add padding to make the image square
    original_size = control_image.size
    padded_control_image, pad_info = add_padding_to_square(control_image, target_size=1024, pad_color='black')

    # Generate images based on num_generations
    outputs = []
    for i in range(num_generations):
        generated_image = pipe(
            prompt,
            control_image=padded_control_image,
            control_mode=control_mode,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            joint_attention_kwargs={"scale": lora_scale}
        ).images[0]

        # Crop the generated image back to the original aspect ratio
        final_image = crop_to_original_ratio(generated_image, original_size, pad_info)

        # Save generated images to /workspace
        image_path = f"/workspace/output_image_{i + 1}.jpg"
        final_image.save(image_path)

        # Read the saved image and convert to Base64
        image_base64 = file_to_base64(image_path)
        outputs.append(image_base64)

    return {"outputs": outputs}  # Return Base64-encoded images

# Start the serverless handler
runpod.serverless.start({"handler": handler})
