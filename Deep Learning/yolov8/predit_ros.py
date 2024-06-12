import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO #instalar
import cv2 #instalar

model = YOLO("D:/yolov8/runs/classify/train3/weights/best.pt")  # load a custom model

def callback(img):
    global model
    # Load a model
    #model = YOLO("/home/sailab/Documents/uxue/yolov8/runs/classify/train/weights/best.pt")  # pretrained YOLOv8n model
    read_image = cv2.imdecode(img.data)
    # Run batched inference on a list of images
    results = model(read_image)  # return a list of Results objects
    #results = model(["c:/Users/uxuai/Downloads/prueba.jpg"])  # return a list of Results objects

def listener():
    rospy.init_node('image_inference', anonymous=True)
    rospy.Subscriber("/camera_image", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
