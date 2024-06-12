import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Cargar el modelo YOLOv8 una vez al inicio
model = YOLO("/home/turtlepc/turtlebot_ws/src/Deep/best.pt")

# Crear una instancia de CvBridge
bridge = CvBridge()

def callback(img):
    global model
    try:
        # Convertir la imagen ROS a una imagen OpenCV
        read_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return
    
    # Realizar la inferencia con el modelo
    results = model(read_image)  # return a list of Results objects
    
    # Extraer y formatear la información que te interesa
    if results:
        result = results[0]  # Considerando solo el primer resultado
        labels_probs = result.probs.topk(5)  # Obtener las top 5 clases con mayor probabilidad
        names = result.names
        for label, prob in zip(labels_probs[1], labels_probs[0]):
            rospy.loginfo("Label: {}, Probability: {:.2f}".format(names[int(label)], prob))

def listener():
    rospy.init_node('image_inference', anonymous=True)
    rospy.Subscriber("/camera_image", Image, callback)

    # spin() simplemente mantiene el script en ejecución hasta que se detenga el nodo
    rospy.spin()

if __name__ == '__main__':
    listener()
