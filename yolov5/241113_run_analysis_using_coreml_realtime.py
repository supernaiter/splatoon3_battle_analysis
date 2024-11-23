#%%
import warnings
# Deprecation warningを抑制
warnings.filterwarnings('ignore', category=FutureWarning)

import datetime
from collections import deque
import csv
import random
import glob
import numpy as np
import cv2
import statistics
import os
from PIL import Image
import time
import re
import coremltools as ct
import torch

#%%
# Core MLモデルのロード
def load_coreml_models():
    model = ct.models.MLModel('../models/the_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    ocr_model = ct.models.MLModel('../models/ocr_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    message_ocr_model = ct.models.MLModel('../models/message_ocr_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    return model, ocr_model, message_ocr_model

# Core ML用の推論関数
def inference_frame(model, frame):
    # 入力画像の前処理
    input_image = cv2.resize(frame, (640, 640))
    # OpenCV(BGR)からPIL(RGB)に変換
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)
    
    # 推論実行
    input_dict = {'input_image': input_image}
    results = model.predict(input_dict)
    
    # デバッグ出力
    print("Core ML output format:")
    print(results)
    
    return standardize_results(results)

# Core MLの出力を標準化する関数
def standardize_results(coreml_output):
    # Core MLの出力から検出結果を取得
    output_array = coreml_output['var_833'][0]  # 最初の次元を削除
    
    # 検出結果を変換
    detections = []
    for detection in output_array:
        # 最初の4つの値がバウンディングボックスの座標
        x1, y1, x2, y2 = detection[:4]
        # 残りの値から最大の確信度とそのクラスインデックスを取得
        confidence_values = detection[4:]
        class_id = np.argmax(confidence_values)
        confidence = confidence_values[class_id]
        
        if confidence > 0.25:  # 確信度のしきい値
            detections.append([x1, y1, x2, y2, confidence, class_id])
    
    # numpy配列に変換
    detections = np.array(detections) if detections else np.zeros((0, 6))
    
    # PyTorch YOLOv5の出力形式に合わせたオブジェクトを作成
    class Results:
        def __init__(self, detections, original_img):
            self.xyxy = [torch.from_numpy(detections)]
            self.ims = [original_img]
            self._names = None
        
        @property
        def names(self):
            if self._names is None:
                self._names = {
                    0: 'moving_count',
                    1: 'fixed_count',
                    2: 'penalty',
                    3: 'message',
                    4: 'alive',
                    5: 'dead',
                    6: 'special',
                    7: 'hoko_canmon',
                    8: 'yagura_kanmon',
                    9: 'area_object',
                    10: 'asari_object',
                    11: 'player'
                }
            return self._names
        
        def print(self):
            for det in self.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det
                print(f"Class: {self.names[int(cls)]}, Confidence: {conf:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    return Results(detections, coreml_output.get('input_image'))

#%%
# メインのリアルタイム処理ループ
def main():
    # モデルのロード
    model, ocr_model, message_ocr_model = load_coreml_models()
    
    # 動画ファイルのリスト取得
    video_files = glob.glob('../footages/*.mp4')
    print(f"Found {len(video_files)} video files")
    random.shuffle(video_files)

    for input_video_path in video_files:
        print(f"Processing: {input_video_path}")
        csv_path = input_video_path.split(".")[0] + "_coreml.csv"
        
        if os.path.isfile(csv_path):
            print(f"Skipping: {csv_path} already exists")
            continue

        # ビデオキャプチャの設定
        cap = cv2.VideoCapture(input_video_path)
        fps = 5
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // fps)
        frame_count = 0
        final_result = deque()

        while True:
            count = 0
            while count < frame_interval:
                ret = cap.grab()
                if not ret:
                    break
                count += 1

            if not ret:
                break

            # フレーム読み込みと処理時間計測
            total_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 推論実行
            results = inference_frame(model, frame)
            
            # プレビュー表示（デバッグ用）
            preview_frame = frame.copy()
            # 検出結果の描画処理を追加
            
            preview_scale = 0.5
            preview_frame_resized = cv2.resize(preview_frame, None, 
                                             fx=preview_scale, 
                                             fy=preview_scale)
            cv2.imshow("Core ML Detection", preview_frame_resized)

            # 処理時間の計算と表示
            total_time = time.time() - total_start
            print(f"\nFrame {frame_count}:")
            print(f"Total processing time: {total_time*1000:.1f}ms")
            print(f"FPS: {1/total_time:.1f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 結果の保存
        final_result = list(final_result)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for res in final_result:
                writer.writerow(res)

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 