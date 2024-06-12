from ultralytics import YOLO

# Load a model

model = YOLO("yolov8x-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/sailab/Documents/uxue/yolov8/50", epochs=100, imgsz=224, patience=20)

