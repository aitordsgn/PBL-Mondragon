import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

# Cambia el path al que necesites, donde tengas tu modelo
model = YOLO("/home/turtlepc/turtlebot_ws/src/Deep/best.pt")

# CvBridge para convertir imágenes ROS a OpenCV
bridge = CvBridge()

def callback(img_msg):
    global model
    try:
        # Convertir el mensaje de imagen ROS a una imagen OpenCV
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Ejecutar la inferencia del modelo
    results = model(cv_image)

    # Procesar y visualizar los resultados
    annotated_image = cv_image.copy()
    for result in results:
        # Cada resultado contiene los atributos necesarios
        for box in result.boxes:
            # Extraer las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            label = model.names[class_id]

            # Dibujar la caja delimitadora y el texto en la imagen
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen con las anotaciones (opcional)
    cv2.imshow("YOLO Detections", annotated_image)
    cv2.waitKey(1)

def listener():
    rospy.init_node('image_inference', anonymous=True)
    rospy.Subscriber("/camera/image", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

