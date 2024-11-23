import sys
import os
import warnings

import torch
from torch import serialization
from models.yolo import DetectionModel
import coremltools as ct

# DetectionModelとsetを安全なグローバルとして登録
serialization.add_safe_globals([DetectionModel, set])

# モデルのエクスポート
try:
    checkpoint = torch.load('models/the_model.pt', 
                          map_location=torch.device('cpu'),
                          weights_only=True)
except Exception as e:
    print("weights_onlyでの読み込みに失敗しました。安全でない方法で読み込みを試みます。")
    checkpoint = torch.load('models/the_model.pt', 
                          map_location=torch.device('cpu'),
                          weights_only=False)

if isinstance(checkpoint, dict):
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint['ema'].model
else:
    model = checkpoint

# FP16からFP32に変換
model = model.float()

# モデルを評価モードに設定
model.eval()

# warningsを無視
warnings.filterwarnings('ignore')

# サンプル入力を作成
example_input = torch.rand(1, 3, 640, 640)

# TorchScriptモデルに変換（簡略化されたフォワードパスを使用）
class TracingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x, augment=False)[0]  # 推論モードのみの出力

wrapped_model = TracingWrapper(model)

try:
    print("トレースモードでの変換を試みます...")
    traced_model = torch.jit.trace(wrapped_model, example_input)
except Exception as e:
    print(f"トレースに失敗しました: {e}")
    print("scriptモードでの変換を試みます...")
    traced_model = torch.jit.script(wrapped_model)

# Core ML変換
ct_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_1", shape=(1, 3, 640, 640))],
    compute_units=ct.ComputeUnit.ALL,
    source="pytorch",
    minimum_deployment_target=ct.target.iOS15
)
print(ct_model)
# 保存
ct_model.save("the_model.mlmodel")
