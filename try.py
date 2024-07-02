import ray
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
import os

# 初始化Ray
ray.init()

# 读取CSV文件
ds = ray.data.read_csv("try.csv")

# 定义图像生成函数
def text2image(row):
    prompt = row["Prompt"]
    image_path = f"images/{prompt.replace(' ', '_')}.png"  # 生成图像文件的路径
    # 本地模型路径
    local_model_path = "./stable-diffusion-v1-4"
    
    # 设定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 从本地加载 Stable Diffusion 模型
    pipe = StableDiffusionPipeline.from_pretrained(local_model_path).to(device)
    
    # 生成图像
    image = pipe(prompt=prompt).images[0]
    
    # 确保保存路径存在
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    # 保存图像
    image.save(image_path)
    
    # 返回包含新列的字典，其中 image 是图像路径
    return {**row, "Image": image_path}

# 应用图像生成函数并添加新列
ds = ds.map(text2image)

# 将结果保存回CSV文件
df = ds.to_pandas()
df.to_csv("try_with_images.csv", index=False)

# 关闭Ray
ray.shutdown()
