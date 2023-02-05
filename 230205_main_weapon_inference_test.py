#%%
from __future__ import print_function, division
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)
    
mean = (0.5,)
std = (0.5,)

transform = ImageTransform(mean, std)

f = open("main_weapon_list.txt","r")
class_names= []

for x in f:
    class_names.append(x.rstrip("\n"))
    #以下のようにしてしまうと、改行コードがlistに入ってしまうため注意
    #list_row.append(x)
f.close()

img = Image.open("52-Gal.jpg")

inputs = transform(img)
inputs = inputs.unsqueeze(0).to(device)

model = torch.load('main_weapons_classification_weight.pth')
model.eval()  ## torch.nn.Module.eval

with torch.no_grad():
    outputs = model(inputs)
    batch_probs = F.softmax(outputs, dim=1)
    batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)

    for probs, indices in zip(batch_probs, batch_indices):
        for k in range(3):
            print(f"Top-{k + 1} {class_names[indices[k]]} {probs[k]:.2%}")
# %%
