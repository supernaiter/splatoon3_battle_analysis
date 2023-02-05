#%%
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weapon_model = torch.load('main_weapons_classification_weight.pth')

class ImageTransform:
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)
mean = (0.5,)
std = (0.5,)
transform = ImageTransform(mean, std)



with open("main_weapon_list.txt", "r") as f:
    class_names = [x.rstrip("\n") for x in f]



def classify_image(img_path, model, k=3):
    img = Image.open(img_path)
    inputs = transform(img)
    inputs = inputs.unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        batch_probs = F.softmax(outputs, dim=1)
        batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
        for probs, indices in zip(batch_probs, batch_indices):
            for i in range(k):
                print(i)
                print(indices[i])
                print(class_names[indices[i]])

#%%
classify_image("dinamo.jpg", weapon_model, k=3)
#%%
classify_image("sharpmarker.jpg", weapon_model, k=3)

# %%
