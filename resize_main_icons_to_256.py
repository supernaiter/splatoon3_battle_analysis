import os
from PIL import Image

weapons_dir = "main_icons"
for filename in os.listdir(weapons_dir):
    if filename.endswith(".png"):
        filepath = os.path.join(weapons_dir, filename)
        img = Image.open(filepath)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(filepath)