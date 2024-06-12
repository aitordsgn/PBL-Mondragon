from ultralytics import YOLO

# Load a model
#model = YOLO("/home/sailab/Documents/uxue/yolov8/runs/classify/train/weights/best.pt")  # pretrained YOLOv8n model
model = YOLO("C:/Users/uxuai/Desktop/yolov8/runs/classify/results2/train_l/weights/best.pt")  # load a custom model

# Run batched inference on a list of images
results = model(["C:/Users/uxuai/Desktop/yolov8/img_prueba/tunnel.jpg"])  # return a list of Results objects
for result in results:
    result.show()  # display to screen
    result.save(filename="result_tunnel.jpg")  # save to disk