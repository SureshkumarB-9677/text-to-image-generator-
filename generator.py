!pip install -q -U diffusers transformers ftfy
!pip install -q -U "ipywidgets>=7,<8"

import torch
from diffusers import StableDiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)


def generate_image(prompt):
 
  image = pipe(prompt).images[0]
  return image


prompt = "A majestic dragon sitting in the forest"
image = generate_image(prompt)


display(image)