'''
Compare the dinno similarity of the reference and the edit images
'''
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import torch.nn as nn

import glob
import numpy as np
import matplotlib.pyplot as plt

def similarity(image1, image2):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)

    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()
    sim = (sim+1)/2
    return sim

def plotFile(data):
    plt.figure(figsize=(10, 6), dpi=300)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    plt.errorbar(range(len(means)), means, yerr=stds, fmt='o', capsize=5, linestyle='none')
    plt.axhline(y=0.95, color='g', linestyle='--')

    # 添加标签和标题
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Value')
    plt.title('Layer Vitality')
    plt.savefig('./output/layer_vitality.png')


"""
sim_list = []
for i in range(64):
    i += 1
    reference = glob.glob(f'./output/ref/{i}.jpg')[0]
    single_sim = []
    
    for j in range(57):
        layer = glob.glob(f'./output/vital2/{i}_{j}.jpg')[0]

        image1 = Image.open(reference)
        image2 = Image.open(layer)
        sim = similarity(image1, image2)
        print(f'The similarity of {i}_{j} is {sim}')
        single_sim.append(sim)
    sim_list.append(single_sim)

sim_list = np.array(sim_list)
np.save(f'./output/sim_list.npy', sim_list)
"""

sim_list = np.load(f'./output/sim_list.npy')
plotFile(sim_list)
