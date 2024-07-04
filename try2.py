from typing import Dict
import pandas as pd
import ray
import torch
from diffusers import StableDiffusionPipeline
import os

# Initialize Ray
ray.init()

# Define a class for the image prediction using the Stable Diffusion model
class ImagePredictor:
    def __init__(self):
        # Define the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model from a local directory
        self.pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4").to(self.device)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Process each prompt in the batch
        batch['Image'] = batch['Prompt'].apply(self.generate_image)
        return batch

    def generate_image(self, prompt: str) -> str:
        # Generate image path
        image_path = f"images/{prompt.replace(' ', '_')}.png"
        # Generate image
        image = self.pipe(prompt=prompt).images[0]
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Save the image
        image.save(image_path)
        return image_path

# Read CSV file
ds = ray.data.read_csv("try.csv")

# Create an instance of the ImagePredictor
predictor = ImagePredictor()

# Apply the image generation function and add a new column using map_batches
ds = ds.map_batches(
    predictor,
    concurrency=2,
    batch_size=4, 
    num_gpus=1,
    batch_format="pandas",
)

# Save the results back to a CSV file
df = ds.to_pandas()
df.to_csv("try_with_images.csv", index=False)

# Shutdown Ray
ray.shutdown()
