import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os

def generate_text2img(prompt, device="mps"):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Generate image
    image = pipe(prompt).images[0]

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"assets/outputs/text2img_{timestamp}.png"
    os.makedirs("assets/outputs", exist_ok=True)
    image.save(out_path)

    return out_path
