import time
import rospy
import numpy as np
import math
import tf
from enum import Enum
from std_msgs.msg import UInt8, Float64
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from turtlebot3_autorace_msgs.msg import MovingParam
#from dqn_Imp import QNet
#from dqn_Imp import RL_agent
from geometry_msgs.msg import Twist
import rospy
from std_msgs.msg import Int32
import rospy
from geometry_msgs.msg import Twist
import rospy
from geometry_msgs.msg import Twist
from time import sleep
import math
PI = math.pi

class LOGIC():
    def __init__(self):
        rospy.init_node('control_moving')
        
        rospy.Subscriber('/sign', Int32, self.callbacksign)
        rospy.Subscriber('/cmd_velsign', Twist, self.callbackcmd)

        # Publishers
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.paused = False
        self.school = False

        rospy.on_shutdown(self.fnShutDown)

    def action_for_Ceda(self):
        rospy.loginfo("Action for Ceda executed.")
        self.publish_twist_for_duration(3)

    def publish_twist_for_duration(self,duration):
        # Crea el mensaje Twist
        twist = Twist()
        # Since we are moving just in x-axis
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        
        # Obtiene el tiempo inicial
        start_time = rospy.Time.now().to_sec()
        
        #rate = rospy.Rate(10)  # 10 Hz

        # Publica el mensaje durante el tiempo especificado
        while rospy.Time.now().to_sec() - start_time < duration:
            self.pub_cmd_vel.publish(twist)
            sleep(0.01)  # Add a small delay to avoid overwhelming the communication
        #print('fin')

    def action_for_TurnLeft(self):
        rospy.loginfo("Action for TurnLeft executed.")
        self.move(0.1,0.5,1)
        self.rotate(10,90,0)

    def action_for_TurnRight(self):
        rospy.loginfo("Action for TurnRight executed.")
        self.move(0.1,0.5,1)
        self.rotate(10,90,1)

    def action_for_Prohibited(self):
        rospy.loginfo("Action for Prohibited executed.")
        self.publish_twist_message_prohibited()

    def publish_twist_message_prohibited(self):
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        
        rospy.loginfo("Publishing twist message and waiting for keyboard input (press any key to continue)...")
        
        # Publicar el mensaje
        self.pub_cmd_vel.publish(twist)
        
        # Esperar hasta que se reciba la entrada del teclado
        self.pause_ros_until_input()

    def pause_ros(self):
        rospy.loginfo("ROS execution paused. Press any key to continue.")
        self.paused = True

    def resume_ros(self):
        rospy.loginfo("Resuming ROS execution.")
        self.paused = False

    def pause_ros_until_input(self):
        self.pause_ros()
        value = input("Please enter any key to continue: ")
        self.resume_ros()

    def action_for_School(self):
        rospy.loginfo("Action for School executed.")
        # We have to call the class from Jon Peder Ros
        self.listener_and_publisher_school()
        pass

    # def callbackcmdschool(self, data):
    #     if not self.paused:
    #         if not self.paused:
    #             #rospy.loginfo("Recibido mensaje de /control/cmd_velsign")
    #             self.pub_cmd_vel.publish(data - 0.05)
    #             #rospy.loginfo("Publicado mensaje en /cmd_vel")
    #         else:
    #             print('stop')
    #             #rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU
    #             #rospy.loginfo("Recibido mensaje de /control/cmd_velsign")
    #             self.pub_cmd_vel.publish(data - 0.05)
    #             #rospy.loginfo("Publicado mensaje en /cmd_vel")
    #     else:
    #         print('stop')
    #         #rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU
        
    def pause_school(self):
        rospy.loginfo("ROS execution paused. Press any key to continue.")
        self.school = True

    def resume_school(self):
        rospy.loginfo("Resuming ROS execution.")
        self.school = False

    def pause_ros_until_input_school(self):
        self.pause_school()
        value = input("Please enter any key to continue: ")
        self.resume_school()

    def listener_and_publisher_school(self):
        # This should not call rospy.init_node again
        # rospy.init_node('cmd_vel_relay', anonymous=True)
        self.pause_ros_until_input_school()
        # Suscribirse al tópico /control/cmd_velsign
        

    def action_for_Tunnel(self):
        rospy.loginfo("Action for Tunnel executed.")
        self.move(0.1,1,1)

    def action_for_Parking(self):
        agente = RL_agent()
        done = agente.RL_guidance(1)

        rospy.loginfo("Action for Parking executed.")
        # We have to call the class from Aitor
        pass

    def move(self, speed, distance, isForward):
        # Starts a new node
        vel_msg = Twist()

        # Checking if the movement is forward or backwards
        if isForward == 1:
            vel_msg.linear.x = abs(speed)
            print('delante')
        else:
            vel_msg.linear.x = -abs(speed)
            print('atras')
            
        # Since we are moving just in x-axis
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0

        # Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        current_distance = 0

        # Loop to move the turtle in a specified distance
        while current_distance < distance:
            # Publish the velocity
            self.pub_cmd_vel.publish(vel_msg)
            # Takes actual time to velocity calculus
            t1 = rospy.Time.now().to_sec()
            # Calculates distancePoseStamped
            current_distance = speed * (t1 - t0)
            #print(current_distance)
            sleep(0.01)  # Add a small delay to avoid overwhelming the communication
            
        # After the loop, stops the robot
        vel_msg.linear.x = 0
        # Force the robot to stop
        self.pub_cmd_vel.publish(vel_msg)
        sleep(0.01)  # Add a small delay to avoid overwhelming the communication
        #self.fnShutDown()
        #print(vel_msg)
        #print('stoppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp')


    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.pub_cmd_vel.publish(twist)
        sleep(0.01)  # Add a small delay to avoid overwhelming the communication

    def callbackcmd(self, data):
        if not self.paused:
            if not self.school:
                 #rospy.loginfo("Recibido mensaje de /control/cmd_velsign")
                 self.pub_cmd_vel.publish(data)
                 #rospy.loginfo("Publicado mensaje en /cmd_vel")
            else:
                 print('lento')
                 #rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU
                 #rospy.loginfo("Recibido mensaje de /control/cmd_velsign")
                 data.linear.x=data.linear.x - 0.07
                 self.pub_cmd_vel.publish(data)
                 #rospy.loginfo("Publicado mensaje en /cmd_v
        else:
            print('stop')
            #rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU
        



    def rotate(self, speed, angle, clockwise):
        # Starts a new node
        vel_msg = Twist()

        # Converting from angles to radians
        angular_speed = speed * 2 * PI / 360
        relative_angle = angle * 2 * PI / 360

        # We won't use linear components
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        # Checking if our movement is CW or CCW
        if clockwise == 1:
            vel_msg.angular.z = -abs(angular_speed)
        else:
            vel_msg.angular.z = abs(angular_speed)

        # Setting the current time for distance calculation
        t0 = rospy.Time.now().to_sec()
        current_angle = 0

        while current_angle < relative_angle:
            self.pub_cmd_vel.publish(vel_msg)
            t1 = rospy.Time.now().to_sec()
            current_angle = angular_speed * (t1 - t0)
            sleep(0.01)  # Add a small delay to avoid overwhelming the communication

        # Forcing our robot to stop
        vel_msg.angular.z = 0
        self.pub_cmd_vel.publish(vel_msg)
        sleep(0.01)  # Add a small delay to avoid overwhelming the communication


    def callbacksign(self, data):
        if not self.paused:
            valor_recibido = data.data
            rospy.loginfo(f"Valor recibido: {valor_recibido}")
            # Aquí puedes poner el código que deseas ejecutar con el valor recibido
            self.procesar_valor(valor_recibido)
        else:
            print('stop')
            #rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU

        

    def procesar_valor(self, valor):
        #print('valor')
        #print(valor)
        if valor == 0:
            self.listener_and_publisher()
        elif valor == 1:
            self.action_for_Parking()
        elif valor == 2:
            self.action_for_Tunnel()
        elif valor == 3:
            self.action_for_School()
        elif valor == 4:
            self.action_for_Prohibited()
        elif valor == 5:
            self.action_for_TurnRight()
        elif valor == 6:
            self.action_for_TurnLeft()
        elif valor == 7:
            self.action_for_Ceda()
        else:
            rospy.logwarn(f"Valor no esperado: {valor}")

    def listener(self):
        # rospy.init_node should be called only once, in the __init__ method
        # rospy.init_node('subscriptor_sign', anonymous=True)
        # Iniciamos el bucle de ROS
        while not rospy.is_shutdown():
            rospy.sleep(0.1)  # Dormimos brevemente para evitar el uso excesivo de la CPU
        
        self.fnShutDown()

    def main(self):
        self.listener()
        rospy.spin()
         

if __name__ == '__main__':
    node = LOGIC()
    node.main()
