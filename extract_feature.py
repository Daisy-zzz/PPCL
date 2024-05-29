'''
CLIP feature
'''
import torch
from transformers import CLIPModel, CLIPProcessor
import os
import pandas as pd
import Image
import numpy as np

device = torch.device("cuda")
image_features = []
text_features = []
model = CLIPModel.from_pretrained("./clip-vit-base-patch32").to(device)
# for name, param in model.named_parameters():
#     print(name, param.size())
processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
batch = 512
data = pd.read_csv("./dataset/train_data_preprocessed.csv")

a = int(len(data) / batch)
b = len(data) - a * batch

for j in range(0, a):
    print(j)
    images = []
    texts = []
    for i in range(j * batch, (j + 1) * batch):
        text = 'Title is ' + data['title'][i] + '. Three-level categories are ' + ' '.join([data['category'][i], data['subcategory'][i], data['concept'][i]]) + '. Tags are ' + data['tags'][i]
        image_path = "./dataset/train_images/" + data['vuid'][i] + '.jpg'
        image = Image.open(image_path).convert("RGB")
       
        images.append(image)
        texts.append(text)
    inputs = processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_feature = outputs.image_embeds
    text_feature = outputs.text_embeds
    image_features.append(image_feature.cpu().numpy())
    text_features.append(text_feature.cpu().numpy())
#
images = []
texts = []
for i in range(a * batch, a * batch + b):
    print(i)
    text = 'Title is ' + data['title'][i] + '. Three-level categories are ' + ' '.join([data['category'][i], data['subcategory'][i], data['concept'][i]]) + '. Tags are ' + data['tags'][i]
    image_path = "./dataset/train_images/" + data['vuid'][i] + '.jpg'
    if not os.path.exists(image_path):
        print(image_path)
        continue
    image = Image.open(image_path).convert("RGB")
    images.append(image)
    texts.append(text)
inputs = processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True).to(device)
with torch.no_grad():
    outputs = model(**inputs)
image_feature = outputs.image_embeds
text_feature = outputs.text_embeds
image_features.append(image_feature.cpu().numpy())
text_features.append(text_feature.cpu().numpy())

n = np.vstack(text_features)
np.save("./dataset/features/text_clip.npy", n)
m = np.vstack(image_features)
np.save("./dataset/features/image_clip.npy", m)