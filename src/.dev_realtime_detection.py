import sys
import os
import argparse
import datetime
from collections import deque
import csv
import random
import glob
import torch
import numpy as np
import cv2
import statistics
import warnings
import platform
import re
from PIL import Image
import time

# 警告を無視
warnings.filterwarnings("ignore")

# パスの設定
yolov5_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov5')
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, yolov5_path)
sys.path.insert(0, src_path)

# デバイスの設定
device = torch.device("mps") if platform.system() == 'Darwin' else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 必要なモジュールのインポート
from src.utils.camera_utils import select_camera, initialize_camera
from src.utils.weapon_utils import (
    ImageTransform, 
    load_weapon_model, 
    load_weapon_list,
    output_weapon_names_pytorch,
    pytorch_weapon_classification
)

# グローバルスコープに移動
message_dict = {
    0: "1", 1: "2", 2: "3", 3: "ア", 4: "ば", 5: "バ", 6: "チ", 7: "ちゅう",
    8: "だ", 9: "第", 10: "ど", 11: "ド", 12: "エ", 13: "防", 14: "が",
    15: "ガ", 16: "グ", 17: "保", 18: "ホ", 19: "い", 20: "カ", 21: "確",
    22: "け", 23: "こ", 24: "コ", 25: "みな", 26: "も", 27: "モ", 28: "ン",
    29: "に", 30: "お", 31: "おう", 32: "破", 33: "プ", 34: "ラ", 35: "れ",
    36: "リ", 37: "る", 38: "さ", 39: "し", 40: "ス", 41: "スー", 42: "た",
    43: "ト", 44: "突", 45: "到", 46: "っ", 47: "ツ", 48: "着", 49: "う",
    50: "ウ", 51: "失", 52: "わ", 53: "を", 54: "ヲ", 55: "ヤ"
}

def get_image(results, num):
    """特定のクラス番号の検出結果から画像を切り出す"""
    return_array = None
    np_results = results.xyxy[0].cpu().numpy()
    array = np_results[np_results[:, 5] == num]
    if len(array) > 0:
        img = results.ims[0][int(array[0][1]):int(array[0][3]), 
                           int(array[0][0]):int(array[0][2])]
        return_array = img
    return return_array

def ocr_using_yolo(ocr_model, count_img):
    """YOLOv5を使用してOCR処理を行う"""
    ocr_results = ocr_model(count_img, 64)
    numbers_list = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nokori']
    output = ""
    
    np_ocr_results = ocr_results.xyxy[0].cpu().numpy()
    ocr_array = np_ocr_results[np_ocr_results[:, 5] < 11]
    ocr_array = ocr_array[np.argsort(ocr_array[:, 0])]
    
    for arr in ocr_array:
        output += str(numbers_list[int(arr[5])])
    
    return output

def output_count_numbers(results, ocr_model, center, moving_count_class_num, fixed_count_class_num):
    """カウント数を検出して出力"""
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    
    moving_count_array = np_results[np_results[:, 5] == moving_count_class_num]
    fixed_count_array = np_results[np_results[:, 5] == fixed_count_class_num]
    count_array = np.concatenate([moving_count_array, fixed_count_array])
    count_array = count_array[np.argsort(count_array[:, 0])]
    
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                               int(count_array[i][0]):int(count_array[i][2])]
            
            OCR_value = ocr_using_yolo(ocr_model, img)
            try:
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if (count_array[i][0] + count_array[i][2]) / 2 < center:
                    if 0 < OCR_result < 101:
                        count_list[0] = OCR_result
                elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                    if 0 < OCR_result < 101:
                        count_list[1] = OCR_result
            except:
                pass
    
    return count_list

def output_penalty_numbers(results, ocr_model, center, penalty_class_num):
    """ペナルティ数を検出して出力"""
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    
    count_array = np_results[np_results[:, 5] == penalty_class_num]
    count_array = count_array[np.argsort(count_array[:, 0])]
    
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                               int(count_array[i][0]):int(count_array[i][2])]
            
            try:
                OCR_value = ocr_using_yolo(ocr_model, img)
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if 0 < OCR_result < 100:
                    if (count_array[i][0] + count_array[i][2]) / 2 < center:
                        count_list[0] = OCR_result
                    elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                        count_list[1] = OCR_result
            except:
                pass
    
    return count_list

def get_center(results, alive_num, dead_num):
    """画面中心を計算"""
    np_results = results.xyxy[0].cpu().numpy()
    alive_array = np_results[np_results[:, 5] == alive_num]
    dead_array = np_results[np_results[:, 5] == dead_num]
    ikalump_array = np.concatenate([alive_array, dead_array])

    if len(ikalump_array) == 8:
        a = np.sum(ikalump_array[:, 0])
        b = np.sum(ikalump_array[:, 2])
        return (a + b) // 16
    else:
        return results.ims[0].shape[1] // 2

def output_ikalump_line(results, alive_num, dead_num):
    """イカの位置情報を出力"""
    np_results = results.xyxy[0].cpu().numpy()
    alive_array = np_results[np_results[:, 5] == alive_num]
    dead_array = np_results[np_results[:, 5] == dead_num]
    ikalump_array = np.concatenate([alive_array, dead_array])
    
    if len(ikalump_array) == 8:
        center = get_center(results, alive_num, dead_num)
        ikalump_state = []
        
        for i in range(len(ikalump_array)):
            x_center = (ikalump_array[i][0] + ikalump_array[i][2]) / 2
            state = {
                'position': 'left' if x_center < center else 'right',
                'status': 'alive' if ikalump_array[i][5] == alive_num else 'dead'
            }
            ikalump_state.append(state)
        
        return ikalump_state
    return []

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='Splatoon3 realtime detection')
    parser.add_argument('-d', '--debug', action='store_true',
                      help='デバッグモード。サンプル画像でテストを実行して終了します')
    return parser.parse_args()

def run_realtime_detection(model, ocr_model, message_ocr_model, weapon_model, main_list, transform):
    """リアルタイム検出のメイン処理"""
    # クラス番号の取得
    results_names = model.names
    moving_count_class_num = [k for k, v in results_names.items() if v == 'moving_count'][0]
    fixed_count_class_num = [k for k, v in results_names.items() if v == 'fixed_count'][0]
    penalty_class_num = [k for k, v in results_names.items() if v == 'penalty'][0]
    message_num = [k for k, v in results_names.items() if v == 'message'][0]
    alive_num = [k for k, v in results_names.items() if v == 'alive'][0]
    dead_num = [k for k, v in results_names.items() if v == 'dead'][0]
    special_num = [k for k, v in results_names.items() if v == 'special'][0]
    hoko_kanmon_num = [k for k, v in results_names.items() if v == 'hoko_canmon'][0]
    yagura_kanmon_num = [k for k, v in results_names.items() if v == 'yagura_kanmon'][0]
    area_object_num = [k for k, v in results_names.items() if v == 'area_object'][0]
    asari_object_num = [k for k, v in results_names.items() if v == 'asari_object'][0]
    player_num = [k for k, v in results_names.items() if v == 'player'][0]

    # カメラの初期化
    selected_camera = select_camera()
    if selected_camera is None:
        print("カメラが選択されませんでした。終了します。")
        return

    try:
        cap = initialize_camera(selected_camera)
    except RuntimeError as e:
        print(e)
        return

    # 基本的な変数の初期化
    frame_count = 0
    analysis_date = datetime.datetime.now()
    warm_up_batch = []
    final_result = deque()
    start_frame = None
    weapon_list = []
    detected_stage = None
    warm_up_frames = 10
    saved_center = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
    fps = cap.get(cv2.CAP_PROP_FPS)

    csv_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

    while True:
        ret, frame = cap.read()
        if not ret:
            print('カメラからの読み取りに失敗')
            break
            
        frame_count += 1
        
        # 表示用のフレームをコピー
        display_frame = frame.copy()
        
        # RGB変換とYOLO処理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, 640)
        np_results = results.xyxy[0].cpu().numpy()
        
        # 各種カウントの取得
        alive_count = np.sum(np_results[:, 5] == alive_num)
        dead_count = np.sum(np_results[:, 5] == dead_num)
        special_count = np.sum(np_results[:, 5] == special_num)
        area_count = np.sum(np_results[:, 5] == area_object_num)
        asari_count = np.sum(np_results[:, 5] == asari_object_num)
        hoko_count = np.sum(np_results[:, 5] == hoko_kanmon_num)
        yagura_count = np.sum(np_results[:, 5] == yagura_kanmon_num)
        player_count = np.sum(np_results[:, 5] == player_num)
        message_count = np.sum(np_results[:, 5] == message_num)
        
        all_count = alive_count + dead_count + special_count
        result_list = [None] * 33
        result_list[29] = analysis_date

        # フレーム時間の計算
        if start_frame is not None:
            result_list[0] = round((frame_count - start_frame) * (1 / fps), 1)

        # 検出果の描画
        for det in np_results:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"Class {int(cls)} ({conf:.2f})"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # プレイヤー検出時の処理
        if all_count == 8:
            if start_frame is None:
                start_frame = frame_count

            # イカの位置情報の取得
            ikalump_states = output_ikalump_line(results, alive_num, dead_num)
            for i, state in enumerate(ikalump_states):
                result_list[1 + i*3] = state['position']      # 位置
                result_list[2 + i*3] = state['status']        # 生存状態
                result_list[3 + i*3] = state.get('special')   # スペシャル状態

            # ウォームアップ期間中の処理
            if warm_up_frames > 0:
                warm_up_batch.append(results)
                warm_up_frames -= 1
                if warm_up_frames == 0:
                    detected_stage = None
                    weapon_list = pytorch_weapon_classification(
                        warm_up_batch, weapon_model, main_list, device, transform,
                        alive_num, dead_num, special_num
                    )
                    print(result_list)

            if warm_up_frames == 0:
                result_list[21] = detected_stage
                for i in range(len(weapon_list)):
                    result_list[13 + i] = weapon_list[i]

        # オブジェクトカウントの記録
        result_list[22] = asari_count
        result_list[23] = hoko_count
        result_list[24] = area_count
        result_list[25] = yagura_count

        # メッセージの処理
        if message_count > 0:
            message_img = get_image(results, message_num)
            if message_img is not None:
                message = ocr_using_yolo(message_ocr_model, message_img)
                print(message)
                result_list[26] = message

        if player_count > 0:
            result_list[27] = True

        # スコアとペナルティの処理
        center = get_center(results, alive_num, dead_num)
        count_list = output_count_numbers(results, ocr_model, center, 
                                        moving_count_class_num, fixed_count_class_num)
        for i in range(len(count_list)):
            result_list[9 + i] = count_list[i]

        penalty_list = output_penalty_numbers(results, ocr_model, center, penalty_class_num)
        for i in range(len(penalty_list)):
            result_list[11 + i] = penalty_list[i]

        final_result.append(result_list)
        
        # 結果の表示
        display_frame_small = cv2.resize(display_frame, (display_frame.shape[1]//3, display_frame.shape[0]//3))
        cv2.imshow("yolo", display_frame_small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

    # 結果の保存
    final_result = list(final_result)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for res in final_result:
            writer.writerow(res)

def ocr_message_yolo(message_ocr_model, message_img):
    """メッセージのOCR処理を行う"""
    ocr_results = message_ocr_model(message_img, 640)
    output = ""
    col_num = 0
    np_ocr_results = ocr_results.xyxy[0].cpu().numpy()
    ocr_array = np_ocr_results
    ocr_array = ocr_array[np.argsort(ocr_array[:, col_num])]

    for arr in ocr_array:
        output = output + message_dict[int(arr[5])]

    return output

def list_available_cameras():
    """利用可能なカメラの一覧を取得"""
    available_cameras = {}
    
    # 接続されているカメラを順番にチェック
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        ret, _ = cap.read()
        if ret:
            # カメラの情報を取得
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # バックエンド名とデバイス名を取得
            backend = cap.getBackendName()
            name = f"Camera {index}"
            
            # macOSでの詳細なデバイス名取得
            try:
                import subprocess
                cmd = ['system_profiler', 'SPCameraDataType']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output = process.communicate()[0].decode()
                
                for line in output.split('\n'):
                    if f'Camera {index}:' in line or f'Camera #{index}:' in line:
                        next_line = False
                        continue
                    if next_line and ':' in line:
                        name = line.split(':')[1].strip()
                        break
            except:
                pass
            
            available_cameras[index] = {
                'name': name,
                'backend': backend,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'display': f"{name} ({width}x{height} @ {fps}fps) (Backend: {backend})"
            }
        cap.release()
        index += 1
    
    return available_cameras

def main():
    """メイン関数"""
    args = parse_args()
    
    # モデルの初期化
    transform = ImageTransform()
    weapon_model = load_weapon_model('models/main_weapons_classification_weight.pth', device)
    main_list = load_weapon_list("main_weapon_list.txt")
    
    model = torch.hub.load('yolov5', 'custom', path='models/the_model.pt', source='local')
    model.iou = 0.3
    
    ocr_model = torch.hub.load('yolov5', 'custom', path='models/ocr_model.pt', source='local')
    message_ocr_model = torch.hub.load('yolov5', 'custom', path='models/message_ocr_model.pt', source='local')
    
    # クラス番号の取得を前に移動
    results_names = model.names
    alive_num = [k for k, v in results_names.items() if v == 'alive'][0]
    dead_num = [k for k, v in results_names.items() if v == 'dead'][0]
    special_num = [k for k, v in results_names.items() if v == 'special'][0]
    
    if args.debug:
        # デバッグモード
        print("デバッグモードで実行中...")
        img = "sample/battle.png"
        results = model(img, 640)
        results.print()
        
        img = "sample/starting.png"
        results = model(img, 640)
        output_weapon_names_pytorch(
            results, weapon_model, main_list, device, transform,
            alive_num, dead_num, special_num
        )
        return
    
    # 通常モード
    run_realtime_detection(model, ocr_model, message_ocr_model, weapon_model, main_list, transform)

if __name__ == "__main__":
    main()




