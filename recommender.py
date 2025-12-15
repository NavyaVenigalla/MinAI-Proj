import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

DATASET_PATH = "occasion_based"

image_embeddings = []
image_paths = []

def load_images():
    for occasion in os.listdir(DATASET_PATH):
        occasion_path = os.path.join(DATASET_PATH, occasion)
        if not os.path.isdir(occasion_path):
            continue

        for img_name in os.listdir(occasion_path):
            img_path = os.path.join(occasion_path, img_name)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(image)
                    embedding /= embedding.norm(dim=-1, keepdim=True)

                image_embeddings.append(embedding.cpu().numpy())
                image_paths.append(img_path)
            except:
                pass

    return np.vstack(image_embeddings), image_paths


def encode_text(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.cpu().numpy()


def recommend(occasion, top_k=3):
    text_emb = encode_text(occasion)
    similarity = cosine_similarity(text_emb, image_embeddings)
    top_indices = similarity[0].argsort()[-top_k:][::-1]
    return [image_paths[i] for i in top_indices]


# Load embeddings once
image_embeddings, image_paths = load_images()
