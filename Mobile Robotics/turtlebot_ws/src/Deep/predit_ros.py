import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32  # Para publicar el número de la clase
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Cambia el path al que necesites, donde tengas tu modelo --> no hay que descomprimir, quédate con el best.pt
model = YOLO("/home/turtlepc/turtlebot_ws/src/Deep/best.pt")

# CvBridge para convertir imágenes ROS a OpenCV
bridge = CvBridge()

# Diccionario que mapea nombres de clases a sus respectivos números
class_signals= {
    'others': 0,
    'parking': 1,
    'tunnel': 2,
    'school_ahead': 3,
    'no_entry': 4,
    'turn_right': 5,
    'turn_left': 6,
    'give_way': 7
}

def callback(img_msg):
    global model
    # Convertir el mensaje de imagen ROS a una imagen OpenCV
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Ejecutar la inferencia del modelo
    results = model(cv_image)  # Devolver una lista de objetos de Resultados

    # Extraer y formatear la información que te interesa
    if results:
        result = results[0]  # Considerando solo el primer resultado
        names = result.names

        # Obtener la clase con mayor probabilidad y su respectiva confianza
        top1_label = result.probs.top1
        top1_conf = result.probs.top1conf
        label_name = names[int(top1_label)]
        prob = top1_conf.item()

        if label_name == 'turn_right':
            label_name = 'turn_left'
            
        # Obtener el número de la clase del diccionario de mapeo
        label_index = class_signals.get(label_name, 0)

        if prob > 0.85:
            rospy.loginfo("Clase: {}, Probabilidad: {:.2f}".format(label_index, prob))
            pub.publish(label_index)
        else:
            rospy.loginfo("Probabilidad baja, enviando 0")
            pub.publish(0)

def listener():
    global pub
    rospy.init_node('image_inference', anonymous=True)
    
    # Publicador para el tópico /sign
    pub = rospy.Publisher('/sign', Int32, queue_size=10)
    
    # Suscribirse al tópico de la cámara
    rospy.Subscriber("/camera/image", Image, callback)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
