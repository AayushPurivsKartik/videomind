# utils/embedder.py
import clip
import torch
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP ViT-L/14 on {device}...")
model, preprocess = clip.load("ViT-L/14", device=device)
print("CLIP loaded!")

@torch.no_grad()
def get_image_embedding(image_path: str):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    embedding = model.encode_image(image)
    return embedding.float().cpu().numpy().squeeze()

@torch.no_grad()
def get_text_embedding(text: str):
    text_tokens = clip.tokenize([text]).to(device)
    embedding = model.encode_text(text_tokens)
    return embedding.float().cpu().numpy().squeeze()