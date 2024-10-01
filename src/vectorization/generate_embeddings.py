import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import clip


# For images
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def get_image_embedding(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()[0]

# For text


def load_text_model(model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model


def get_text_embedding(text, model):
    return model.encode(text)


def combine_embeddings(text_embedding, image_embedding, alpha=0.5):
    return alpha * text_embedding + (1 - alpha) * image_embedding


# Test for text, image embedding
if __name__ == '__main__':
    text_model = load_text_model()
    image_model, preprocess, device = load_clip_model()

    text = "I am a sentence for which I would like to get its embedding."
    image_path = "./sample/test.png"

    text_embedding = get_text_embedding(text, text_model)
    image_embedding = get_image_embedding(
        image_path, image_model, preprocess, device)
    combined_embedding = combine_embeddings(text_embedding, image_embedding)

    print(combined_embedding)
    print(combined_embedding.shape)
