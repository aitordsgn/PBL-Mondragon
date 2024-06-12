#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32

def solicitar_valor():
    while True:
        try:
            valor = int(input("Por favor, ingrese un valor entero entre 0 y 7: "))
            if 0 <= valor <= 7:
                return valor
            else:
                print("El valor debe estar entre 0 y 7.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número entero.")

def main():
    rospy.init_node('publicador_senal', anonymous=True)
    pub = rospy.Publisher('/sign', Int32, queue_size=10)
    
    while not rospy.is_shutdown():
        valor = solicitar_valor()
        rospy.loginfo(f"Publicando valor: {valor}")
        pub.publish(valor)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
