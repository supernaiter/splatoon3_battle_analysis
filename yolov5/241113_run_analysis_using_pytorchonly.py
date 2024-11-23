import datetime
from collections import deque
import csv
import random
import glob
import torch
import numpy as np
import cv2
import statistics
import os
from PIL import Image
import time
import re
from torchvision import transforms
import torch.nn.functional as F

# デバイス設定
device = torch.device("mps")

# モデルの読み込み
model = torch.hub.load('.', 'custom', path='../models/the_model.pt', source='local')
model.iou = 0.3  # NMS IoU threshold

ocr_model = torch.hub.load('.', 'custom', path='../models/ocr_model.pt', source='local')
message_ocr_model = torch.hub.load('.', 'custom', path='../models/message_ocr_model.pt', source='local')

# 画像変換用クラス
class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

# 変換パラメータの設定
mean = (0.5,)
std = (0.5,)
transform = ImageTransform(mean, std)

# 武器モデルの読み込み
weapon_model = torch.load('../230206_main_weapons_classification_weight.pth', map_location=device)
weapon_model.eval()

# メインの関数群
def output_ikalump_array(results):
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    alive_array = np_results[np_results[:, 5] == alive_num]
    dead_array = np_results[np_results[:, 5] == dead_num]
    special_array = np_results[np_results[:, 5] == special_num]
    
    ikalump_array = np.concatenate([alive_array, dead_array, special_array])
    ikalump_array = ikalump_array[np.argsort(ikalump_array[:, col_num])]
    return ikalump_array


# メインの処理ロジック
def process_video(input_video_path):
    # ... (既存のビデオ処理ロジック)
    pass

# メイン実行部分
if __name__ == "__main__":
    video_files = glob.glob('../footages/*.mp4')
    print(f"Found {len(video_files)} video files")
    random.shuffle(video_files)

    for input_video_path in video_files:
        process_video(input_video_path)
