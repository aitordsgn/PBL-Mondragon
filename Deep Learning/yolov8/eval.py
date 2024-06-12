from ultralytics import YOLO

# Load a model
#model = YOLO("/home/sailab/Documents/uxue/yolov8/runs/classify/train/weights/best.pt")  # load a custom model

model = YOLO("/home/sailab/Documents/uxue/yolov8/runs/classify/train9/weights/best.pt")  # load a custom model
# Validate the model
metrics = model.val(split='test', imgsz=224)  # no arguments needed, dataset and settings remembered
