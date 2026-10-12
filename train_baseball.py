import os
from ultralytics import YOLO
import torch

# 1. 針對 5090 的架構支援
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0' 

# 2. 開啟 Rich 高質感進度條 (這就是你想要的更明顯 tqdm)
os.environ["YOLO_TQDM_RICH"] = "true"

def train():
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
    
    # 換成小一級的 Large 模型
    model = YOLO('yolo26l.pt') 

    yaml_path = "F:/personal project/train_baseball/datalast/baseball.yaml"

    model.train(
        data=yaml_path,
        epochs=300,
        imgsz=1280,           
        batch=10,             
        device=0,             
        workers=12,           
        project='Baseball_Detection',
        name='yolo26l_5090_stable',  # 換個新資料夾名稱
        amp=True,             
        cache='disk',         
        
        # === 穩定版關鍵設定 ===
        optimizer='auto',     # 放棄手動 AdamW，交給系統自動判斷最安全的優化器與學習率
        box=10.0,             # 從 15 降回 10，依然對坐標要求嚴格，但不會引發爆炸
        close_mosaic=15,      
        single_cls=True,      
        patience=50,          
        # ====================
    )

if __name__ == '__main__':
    train()