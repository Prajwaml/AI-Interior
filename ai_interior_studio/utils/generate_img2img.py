import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from datetime import datetime
import os

def generate_img2img(init_image_path, prompt, device="mps", strength=0.7, guidance=7.5):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    init_image = Image.open(init_image_path).convert("RGB")

    images = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance).images
    image = images[0]

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"assets/outputs/img2img_{timestamp}.png"
    os.makedirs("assets/outputs", exist_ok=True)
    image.save(out_path)

    return out_path
