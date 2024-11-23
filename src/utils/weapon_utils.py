import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import statistics
import numpy as np
from typing import List, Optional

class ImageTransform():
    def __init__(self, mean=(0.5,), std=(0.5,)):
        self.data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

def load_weapon_model(model_path, device):
    """武器分類モデルを読み込む"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def load_weapon_list(file_path):
    """武器リストを読み込む"""
    with open(file_path, "r") as f:
        return [x.rstrip("\n") for x in f]

def output_ikalump_array(results, alive_num, dead_num, special_num):
    """イカランプの配列を取得"""
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    alive_array = np_results[np_results[:, 5] == alive_num]
    dead_array = np_results[np_results[:, 5] == dead_num]
    special_array = np_results[np_results[:, 5] == special_num]
    
    ikalump_array = np.concatenate([alive_array, dead_array, special_array])
    ikalump_array = ikalump_array[np.argsort(ikalump_array[:, col_num])]
    return ikalump_array

def output_weapons_images(results, alive_num, dead_num, special_num) -> Optional[List[np.ndarray]]:
    """プレイヤーの武器画像を抽出"""
    ikalump_array = output_ikalump_array(results, alive_num, dead_num, special_num)
    if len(ikalump_array) == 8:
        imgs = []
        img = results.ims[0]
        for i in range(0, 8):
            imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),
                        int(ikalump_array[i][0]):int(ikalump_array[i][2])])
        return imgs
    return None

def output_weapons_images_for_special(results, alive_num, dead_num, special_num) -> List[np.ndarray]:
    """スペシャル用の武器画像を抽出"""
    ikalump_array = output_ikalump_array(results, alive_num, dead_num, special_num)
    imgs = []
    img = results.ims[0]
    for i in range(0, 8):
        imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),
                    int(ikalump_array[i][0]):int(ikalump_array[i][2])])
    return imgs

def output_weapon_names_pytorch(results, weapon_model, main_list, device, transform, alive_num, dead_num, special_num):
    """武器の分類を実行"""
    with torch.no_grad():
        weapon_classification_list = []
        imgs = output_weapons_images(results, alive_num, dead_num, special_num)
        if imgs is None:
            return None
            
        for img in imgs:
            img = Image.fromarray(img)
            inputs = transform(img)
            inputs = inputs.unsqueeze(0).to(device)
            outputs = weapon_model(inputs)
            batch_probs = F.softmax(outputs, dim=1)
            batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
            for probs, indices in zip(batch_probs, batch_indices):
                weapon_classification_list.append(main_list[indices[0]])
    return weapon_classification_list

def pytorch_weapon_classification(warm_up_batch, weapon_model, main_list, device, transform, alive_num, dead_num, special_num):
    """バッチ処理による武器分類"""
    raw_outputs = []
    final_weapon_result = []
    for batch_result in warm_up_batch:
        raw_outputs.append(output_weapon_names_pytorch(
            batch_result, weapon_model, main_list, device, transform, 
            alive_num, dead_num, special_num
        ))
    
    raw_outputs = np.array(raw_outputs)
    for i in range(raw_outputs.shape[1]):
        final_weapon_result.append(statistics.mode(raw_outputs[:,i]))

    return final_weapon_result

def crop_center(pil_img, crop_width, crop_height):
    """画像の中心部分を切り出す"""
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2)) 