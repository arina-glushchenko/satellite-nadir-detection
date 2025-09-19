import numpy as np
from ultralytics import YOLO

def run_yolo_on_image(image: np.ndarray, weights_path: str, conf: float = 0.25, device: str = 'cpu') -> list:
    """Runs YOLOv8n on a numpy image (H,W,3, uint8). Returns list of detections."""
    
    model = YOLO(weights_path)
    try:
        results = model.predict(source=image, conf=conf, device=device, verbose=False)
    except Exception:
        results = model(image, conf=conf, device=device)
    
    if not results:
        return []
    
    res = results[0]
    boxes = getattr(res, 'boxes', None)
    if boxes is None:
        return []
    
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            data = boxes.data.cpu().numpy()
            xyxy = data[:, :4]
            confs = data[:, 4]
            classes = data[:, 5].astype(int)
        except Exception:
            return []
    
    return [{'xyxy': xyxy[i].tolist(), 'conf': float(confs[i]), 'class': int(classes[i])} for i in range(xyxy.shape[0])]