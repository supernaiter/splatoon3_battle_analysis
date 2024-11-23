# YOLOv5/v8からCore MLへの変換
# export.py使用
import torch
import coremltools as ct
from models.yolo import Model
import logging
import sys
from datetime import datetime

# ロガーの設定
def setup_logger():
    # ロガーの作成
    logger = logging.getLogger('model_converter')
    logger.setLevel(logging.DEBUG)
    
    # ファイルハンドラの作成
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)
    
    # コンソールハンドラの作成
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # フォーマッターの作成
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # ハンドラーの追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def convert_yolo_to_coreml(model_path, output_path, logger):
    logger.info(f"Starting conversion of {model_path}")
    
    try:
        # PyTorchモデルをロード
        logger.debug("Loading PyTorch model...")
        model = torch.hub.load('.', 'custom', path=model_path, source='local')
        model.eval()
        
        # モデルを推論モードに設定
        model.model.float()
        model.model.eval()
        
        # エクスポートモードを設定
        if hasattr(model.model, 'export'):
            model.model.export = True
        
        # サンプル入力を作成
        logger.debug("Creating example input...")
        example_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
        
        try:
            # TorchScriptモデルに変換
            logger.info("Converting to TorchScript...")
            with torch.no_grad():
                # モデルをトレースする前に、一度ダミー入力で実行して形状を安定させる
                model(example_input)
                
                # strict=Falseでトレースを実行
                traced_model = torch.jit.trace(model, example_input, strict=False)
                
                # 保存したモデルを読み込み
                ts_model_path = output_path.replace('.mlpackage', '.torchscript.pt')
                traced_model.save(ts_model_path)
                traced_model = torch.jit.load(ts_model_path)
                traced_model.eval()
            
            # Core MLモデルに変換
            logger.info("Converting to Core ML...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.ImageType(
                    name="input_image",
                    shape=example_input.shape,
                    scale=1/255.0,
                    bias=[0, 0, 0],
                    color_layout="RGB"
                )],
                minimum_deployment_target=ct.target.iOS15,
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            
            # モデルを保存
            logger.info(f"Saving model to {output_path}")
            mlmodel.save(output_path)
            logger.info(f"Successfully converted and saved model to {output_path}")
            
        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            logger.exception("Detailed traceback:")
            raise
            
    except Exception as e:
        logger.error(f"Failed to process model: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    # ロガーのセットアップ
    logger = setup_logger()
    logger.info("Starting model conversion process")
    logger.info(f"Conversion started at: {datetime.now()}")
    
    try:
        # メインモデル
        logger.info("Converting main detection model...")
        convert_yolo_to_coreml("../models/the_model.pt", "../models/the_model.mlpackage", logger)
        
        # OCRモデル
        logger.info("\nConverting OCR model...")
        convert_yolo_to_coreml("../models/ocr_model.pt", "../models/ocr_model.mlpackage", logger)
        
        # メッセージOCRモデル
        logger.info("\nConverting message OCR model...")
        convert_yolo_to_coreml("../models/message_ocr_model.pt", "../models/message_ocr_model.mlpackage", logger)
        
        logger.info("All conversions completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to convert models: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Conversion process finished at: {datetime.now()}")