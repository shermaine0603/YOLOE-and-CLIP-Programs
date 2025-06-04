import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("Photo 3.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of an office", "a photo of a person's room", "a photo of an office pantry"]).to(device)

print(text)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

np.set_printoptions(suppress=True, precision=6) 
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]