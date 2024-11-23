import zmq
import json
from datetime import datetime

def main():
    # ZeroMQ設定
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("受信待機中...")

    try:
        while True:
            # データを受信して表示
            data = socket.recv_json()
            
            # -1をNoneに戻す
            data = [x if x != -1 else None for x in data]
            
            # 現在時刻を追加
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"[{current_time}] 受信データ:", data)

    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main() 