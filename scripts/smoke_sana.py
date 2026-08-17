"""Smoke test: load SANA 600M and write baseline images to outputs/00_baseline."""
import torch
from diffusers import SanaPipeline

MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
DEVICE = "cuda:0"

pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

prompts = [
    "a cat sitting on a windowsill",
    "a portrait of a woman, photo",
    "a city street at night",
    "a bowl of fruit on a table",
]

import os
os.makedirs("outputs/00_baseline", exist_ok=True)
for i, p in enumerate(prompts):
    g = torch.Generator(device=DEVICE).manual_seed(42 + i)
    img = pipe(p, num_inference_steps=20, guidance_scale=4.5, generator=g,
               height=512, width=512).images[0]
    img.save(f"outputs/00_baseline/{i}_{p.replace(' ', '_')[:40]}.png")
    print("saved", p)
print("scheduler:", type(pipe.scheduler).__name__)
print("latent ch:", pipe.transformer.config.in_channels,
      "timestep_scale:", getattr(pipe.transformer.config, "timestep_scale", None))
print("vae scaling:", pipe.vae.config.scaling_factor)
print("max vram GiB:", torch.cuda.max_memory_allocated(DEVICE) / 2**30)
