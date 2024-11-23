import cv2
import subprocess
import platform

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
            if platform.system() == 'Darwin':  # Mac OS
                try:
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
                'display': f"{name} ({width}x{height} @ {fps}fps)"
            }
        cap.release()
        index += 1
    
    return available_cameras

def select_camera():
    """カメラを選択するインタラクティブな関数。OBS Camera Extensionを優先"""
    cameras = list_available_cameras()
    
    if not cameras:
        print("利用可能なカメラが見つかりませんでした")
        return None
    
    # OBS Camera Extensionを探す
    obs_camera_index = None
    for idx, info in cameras.items():
        if any(obs_identifier in info['name'].lower() for obs_identifier in ['obs', 'virtual', 'camera extension']):
            obs_camera_index = idx
            break
    
    if obs_camera_index is not None:
        print(f"\nOBS Camera Extensionを自動選択しました: {cameras[obs_camera_index]['display']}")
        return obs_camera_index
    
    # OBS Camera Extensionが見つからない場合は手動選択
    print("\n利用可能なカメラ:")
    for idx, info in cameras.items():
        print(f"{idx}: {info['display']} (Backend: {info['backend']})")
    
    while True:
        try:
            selection = int(input("\nOBS Camera Extensionが見つかりませんでした。カメラ番号を選択してください (Ctrl+Cで終了): "))
            if selection in cameras:
                return selection
            print("無効な選択です。もう一度試してください。")
        except ValueError:
            print("数字を入力してください")
        except KeyboardInterrupt:
            print("\nキャンセルされました")
            return None

def initialize_camera(camera_id, width=1920, height=1080, fps=30):
    """カメラを初期化して設定を適用"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {camera_id} を開けませんでした")
    
    # カメラの設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # 実際に設定された値を確認
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n実際の設定値:")
    print(f"解像度: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    
    return cap 