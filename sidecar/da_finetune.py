"""Fine-tune Delineate-Anything (YOLOv11-seg) on our Tibet+FSDA parcels -> a strengthened parcel
delineation module (the coupled-delineation idea). Then it can be coupled with our DINOv3 classifier."""
import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
DA = "/Users/zhangfeng/D/delineate_anything/DelineateAnything.pt"
DATA = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/yolo_tibet/data.yaml"
m = YOLO(DA)
m.train(data=DATA, epochs=40, imgsz=768, batch=4, device="mps",
        project="/Users/zhangfeng/D/delineate_anything/ft", name="tibet", exist_ok=True,
        verbose=False, plots=False, lr0=0.002, warmup_epochs=2)
print("DONE best:", m.trainer.best, flush=True)
