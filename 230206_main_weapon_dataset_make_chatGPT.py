import glob
import random
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

class_list = []
for path in glob.glob('chill_season_main_icons/*.png'):
    class_list.append(re.findall('/(.*).png', path)[0].split("/")[1])

with open('weapon_list.txt', 'w') as f:
    for x in class_list:
        f.write(str(x) + "\n")

resize_size = (200, 200)
base_images = glob.glob('background_images/*.jpg')

def generate_dataset(weapons_dir_path):
    label_info = []
    base_path = base_images[random.randint(0, len(base_images) - 1)]
    base = Image.open(base_path)
    base = base.resize(resize_size)
    base = base.filter(ImageFilter.BLUR)
    for i in range(0, 1):
        logo_relative_positions = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
        logo_position = (int(100 * logo_relative_positions[0]), int(100 * logo_relative_positions[1]))
        label = random.choice(class_list)
        logo = Image.open(f'{weapons_dir_path}/{label}.png')
        logo_w, logo_h = logo.size
        logo_scale_x = random.uniform(0.5, 1.0)
        logo_scale_y = random.uniform(0.5, 1.0)
        logo = logo.resize((int(logo_w * logo_scale_x), int(logo_h * logo_scale_y)))
        logo = ImageEnhance.Color(logo)
        logo = logo.enhance(random.uniform(0, 1))
        if random.randint(0, 10) < 2:
            logo = logo.rotate(random.randint(-10, 10), expand=True)
        base.paste(logo, logo_position, logo)
    return base

generate_dataset('chill_season_main_icons')
