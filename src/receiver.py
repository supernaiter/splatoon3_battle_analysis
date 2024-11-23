import multiprocessing as mp
from datetime import datetime
import pandas as pd
from protocol import DetectionMessage, create_game_state
import signal
import sys

class BaseReceiver:
    """受信処理の基底クラス"""
    def process_game_state(self, game_state):
        """ゲーム状態を処理する（サブクラスでオーバーライド）"""
        raise NotImplementedError
    
    def process_error(self, error_message):
        """エラーを処理する（サブクラスでオーバーライド）"""
        raise NotImplementedError
    
    def on_stop(self):
        """停止時の処理（サブクラスでオーバーライド）"""
        raise NotImplementedError

class ConsoleReceiver(BaseReceiver):
    """コンソールに出力するレシーバー"""
    def __init__(self):
        self.message_count = 0
    
    def process_game_state(self, game_state):
        self.message_count += 1
        print("\n" + "="*50)
        print(f"Message #{self.message_count}")
        print(f"Timestamp: {datetime.now()}")
        print("-"*50)
        
        # 重要なフィールドを先に表示
        priority_fields = ['elapsed_time', 'count_left', 'count_right', 'message']
        print("Priority Data:")
        for field in priority_fields:
            value = getattr(game_state, field)
            if value is not None:
                print(f"  {field}: {value}")
        
        print("\nOther Data:")
        for field, value in game_state.__dict__.items():
            if value is not None and field not in priority_fields:
                print(f"  {field}: {value}")
    
    def process_error(self, error_message):
        print(f"\n[ERROR] {error_message}")
    
    def on_stop(self):
        print(f"\nReceiver stopping... Total messages received: {self.message_count}")

def signal_handler(signum, frame):
    print("\nSignal received, shutting down...")
    sys.exit(0)

def receiver_process(queue, receiver_class=ConsoleReceiver):
    """受信プロセス"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "="*50)
    print("Receiver process started... Press Ctrl+C to stop")
    print("Waiting for data...")
    print("="*50 + "\n")
    
    receiver = receiver_class()
    
    while True:
        try:
            # タイムアウトを設定して待機
            try:
                data = queue.get(timeout=1.0)  # 1秒でタイムアウト
            except mp.queues.Empty:
                continue  # タイムアウトしたら次のループへ
            
            # リストデータをDetectionMessageに変換
            if isinstance(data, list):
                message = create_game_state(data)
            elif data == "STOP":
                message = DetectionMessage(type="stop", payload=None)
            else:
                message = DetectionMessage(type="error", payload=str(data))
            
            # メッセージタイプに応じた処理
            if message.type == "game_state":
                receiver.process_game_state(message.payload)
            elif message.type == "error":
                receiver.process_error(message.payload)
            elif message.type == "stop":
                receiver.on_stop()
                break
                
        except Exception as e:
            print(f"\n[ERROR] Error in receiver: {e}")
            continue

if __name__ == "__main__":
    # テインプロセスでもシグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    queue = mp.Queue()
    receiver = mp.Process(target=receiver_process, args=(queue,))
    receiver.start()
    
    try:
        # メインプロセスを待機状態にする
        receiver.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        queue.put("STOP")
        receiver.join()
        sys.exit(0) 