import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numpy import array
import numpy
import numpy as np

import math

import os
import signal
import subprocess
import time
import copy
from math import pi

from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan

import rospy
import rosnode
from std_srvs.srv import Empty

from os import path
import random



class Turtlebot3Ros1DisPBLEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gazebo = False
        self.real_robot_initial_pose = None
        self.real_robot_initial_pose_set = False
        print("------ Initialization start ------")
        # We assume that a ROS node has already been created
        # before initialising the environment
        rospy.init_node('Gym_Turtlebot3_Dis_Env_Node', anonymous=True)
       
        # Set connection with Gazebo simulator
        rosnodes = rosnode.get_node_names()
        self.execution_count = 0 #Execution counter
        if "/gazebo" in rosnodes:
            print('A Gazebo node was found! Assuming Gazebo Simulation')
            self.gazebo = True

        else:
            print('No Gazebo node was found! Assuming real robot')
            self.gazebo = False



        self.entity_dir_path = os.path.dirname(os.path.realpath(__file__))
        print ("path1 = ", self.entity_dir_path)
        self.entity_dir_path = self.entity_dir_path.replace(
            'my-gym3/my_gym3/envs',
            'turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models/my_goal')
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        print ("path2 = ", self.entity_path)

        #self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'
        self.goal_displayed = True

        self.last_distance = 0.0
        self.step_cnt = 0
        self.max_step_episode = 500

        self.allowed_min_obstacle_distance = 0.2
        self.goal_distance_acept = 0.2
        self.goal_distance = 0
        

        self.not_goal_count = 0 # Counter for not reaching the goal
        
        """************************************************************
        ** Manually Configured variables
        ************************************************************"""
        self.near_distance = 0.3
        self.near_colission_distance = 0.2
        self.fastest= self.max_step_episode
        self.goal_achieve_cnt = 0
        
        """************************************************************
        ** Environment configuration variables
        ************************************************************"""
        self.stage = 1
        #TODO change goal depending on if it is real or simulation
        if self.gazebo:
            self.goal_pose_x = -1.6
            self.goal_pose_y = -1.6
            self.generate_goal_pose()
        else:
            initial_x, initial_y = self.real_robot_initial_pose
            self.goal_pose_x = initial_x + 1.3
            self.goal_pose_y = initial_y - 1.6



        self.maxt_distance = 3.0

        # class variables
        self._observation_msg = None
        self.scan_msg_ready = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""

        # Initialise subscribers
        
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('imu', Imu, self.imu_callback)        
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)

        # Initialise publishers
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.goal_pose_pub = rospy.Publisher('goal_pose', Pose, queue_size=5)
                     

        # Initialise client
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        #self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        #self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
       
        self.seed()
        """************************************************************
        ** Initialise Gym Environment
        ************************************************************"""
        # set up action-, observation space and reward range
        self.action_size = 5
        action_num = 1
        self.continuous = False
        #self.shape_value = 1
       
        low = -1.5 * np.ones(action_num)
        high = 1.5 * np.ones(action_num)
       
        #self.low = -1.5
        #self.high = 1.5
        if self.continuous:
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            #self.action_space = spaces.Box(self.low, self.high, shape=(self.shape_value,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.action_size)
        action = self.action_size /2
        state = self.take_observation(action)
        self.observation_dims = len(state)
        print("self.observation_dims",self.observation_dims)
       
        self.observation_space = spaces.Box(low=-3.5, high=3.5, shape=(self.observation_dims,), dtype=np.float32)
        print ("self.observation_space ",self.observation_space )
        self.reward_range = (-np.inf, np.inf)
        self.max_angular_vel = 1.5

        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y          
        self.goal_pose_pub.publish(goal_pose)
        self.scan_ranges = []

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def position_relative (self,msg):
        self.intial_relative_pose_x = msg.pose.pose.position.x
        self.intial_relative_pose_y = msg.pose.pose.position.y
    

    def goal_pose_callback(self, msg):
        self.goal_pose_x = msg.position.x
        self.goal_pose_y = msg.position.y

    def imu_callback(self, msg):
        self.last_ang_velo_x = msg.angular_velocity.x
        #print ("imu", self.last_ang_velo_x)

    def odom_callback(self, msg):
        if not self.gazebo and not self.real_robot_initial_pose_set:
                self.real_robot_initial_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                self.real_robot_initial_pose_set = True

        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = round(math.hypot(self.goal_pose_x - self.last_pose_x, self.goal_pose_y - self.last_pose_y),2)
        
        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)
        #Nomralize goal_angle
        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi                      

        elif goal_angle < -math.pi:            
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle
        #print ("goal angle", goal_angle)
   
    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        # Filtrar los valores de scan_ranges para eliminar los valores 0.0
        filtered_scan_ranges = [value for value in self.scan_ranges if value != 0.0]

        # Calcular la distancia mínima de los valores filtrados
        if filtered_scan_ranges:
            self.min_obstacle_distance = round(min(filtered_scan_ranges),2)
            self.min_obstacle_angle = numpy.argmin(filtered_scan_ranges)
        else:
            # Manejar el caso donde todos los valores sean 0.0 (puedes ajustar esto según tus necesidades)
            self.min_obstacle_distance = float('3.5')  
            self.min_obstacle_angle = float('0.0')  
        
        
        self.scan_msg_ready = True

    def reset_simulation(self):
        print ("--------------- Reset -------------- ")
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
            #print("gazebo/reset_simulation service call executed")
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
           
        #req = Empty.Request()
        #while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
        #self.reset_simulation_client.call_async(req)
   
    def take_observation(self, action):
        
        # Take an observation

        self.scan_msg_ready = False

        a = 0
        while self.scan_msg_ready is False:
            a += 1

        observation = copy.deepcopy(self.scan_ranges)
        goal_distance = copy.deepcopy(self.goal_distance)
        goal_angle = copy.deepcopy(self.goal_angle)
        scan_range = []
        # Limit the maximun distance to 3.5 meters and then normalize
        
        #TODO Checkear los valores de el lidar para ver  que no de 0

        for i in range(0,len(observation),20):
        #for i in range(len(observation)):
            if observation[i] == float('Inf'):
                scan_range.append(3.5/3.5)
            elif np.isnan(observation [i]):
                scan_range.append(0)
            elif observation[i] > 3.5:
                scan_range.append(3.5/3.5)
            else:
                scan_range.append(observation[i]/3.5)

        obs = scan_range + [(goal_angle+math.pi)/(2*math.pi), goal_distance/10.0, action]
        return obs

    def setReward(self, state, action): 
        reward = 0
        done = False

        #print("I am ath this distance from the goal:", current_distance)
        if self.last_distance == 0:
            self.last_distance = self.goal_distance
        distance_rate = (self.last_distance - self.goal_distance) / self.last_distance
        self.last_distance = self.goal_distance
       
        #Reward for getting closer to the goal
        if distance_rate >= 0:
            reward = 2
        else:
            reward = -2
        reward -=1
        # Add  fuel cost reward to be able to achieve faster reaching of the goal for each episode iteration.



        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15 #Reduce the linear velocity to achieve better control
        vel_cmd.angular.z = 0.0
        
        # Punishment  for colissioning with an object
        if self.min_obstacle_distance < self.allowed_min_obstacle_distance:
            print("Collision!!")
            reward = -750
            self.cmd_vel_pub.publish(vel_cmd)
            self.not_goal_count += 1
            done = True


        # Reward for achieving the goal
        if self.goal_distance < self.goal_distance_acept:
            print("Goal!!")
            self.step_cnt = 0
            self.cmd_vel_pub.publish(vel_cmd)
            self.generate_goal_pose()
            goal_pose = Pose()
            goal_pose.position.x = self.goal_pose_x
            goal_pose.position.y = self.goal_pose_y          
            if self.goal_displayed == True:
                #self.delete_entity()
                self.goal_displayed = False
            time.sleep(0.5)
            self.goal_pose_pub.publish(goal_pose)
            #self.spawn_entity()
            time.sleep(0.5)
            self.goal_displayed = True
            reward = 2000
            self.not_goal_count = 0

            done = True
            self.goal_achieve_cnt +=1
            print("Goal achievement count = ", self.goal_achieve_cnt)            

        # Penalty for exceeding the maximum number of steps
        if self.step_cnt >= self.max_step_episode:
            self.step_cnt = 0
            print("Max step achieved!!")
            self.cmd_vel_pub.publish(vel_cmd)
            self.generate_goal_pose()
            goal_pose = Pose()
            goal_pose.position.x = self.final_pose_x
            goal_pose.position.y = self.final_pose_x
            if self.goal_displayed == True:
                #self.delete_entity()
                self.goal_displayed = False
            time.sleep(0.5)
            self.goal_pose_pub.publish(goal_pose)
            #self.spawn_entity()
            self.not_goal_count += 1

            time.sleep(0.5)
            self.goal_displayed = True
            done = True

            # Punishment for being too close to an object
            if self.min_obstacle_distance < self.near_colission_distance:
                reward -= 20  # Punishment for being too close to a colission
                print("Near Collision!")
            elif self.min_obstacle_distance < self.near_distance:
                reward -= 5  # Punishment for being too close to an object

        #print ("Reward:",reward)
        return reward, done

    def step(self, action):
        max_angular_vel = 1.5

        # Define la velocidad angular basado en la acción
        if self.continuous == True:
            ang_vel = action[0]*1.0
        else:            
           ang_vel = action[0] * 1.0 if self.continuous else ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15  #Reduce the linear velocity to achieve better control
        vel_cmd.angular.z = ang_vel
        #print("Angular velocity:",ang_vel)
        self.cmd_vel_pub.publish(vel_cmd)
        self.scan_msg_ready = False

        state = self.take_observation(action)
        reward, done = self.setReward(state,action)
        observation = np.asarray(state)
        truncated = 0
        return observation, reward, done, truncated, {}
    
    #TODO change goal depending on if it is real or simulation
    def generate_goal_pose(self):
        if self.gazebo:
            goal_ok = False
            goal_pose_list = [
                [-1.0, -1.0], [-1.0, -1.25], [-1.0, -1.5], 
                [-1.25, -1.0], [-1.25, -1.25], [-1.25, -1.5], 
                [-1.5, -1.0], [-1.5, -1.25], [-1.5, -1.5]
            ]
            index = random.randrange(0, len(goal_pose_list))
            self.goal_pose_x = goal_pose_list[index][0]
            self.goal_pose_y = goal_pose_list[index][1]
        else:
            if self.real_robot_initial_pose_set:
                initial_x, initial_y = self.real_robot_initial_pose
                self.goal_pose_x = initial_x + 1.3
                self.goal_pose_y = initial_y + 1.4
            else:
                self.goal_pose_x = 1.3
                self.goal_pose_y = 1.4
            
        print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

    def reset(self): 
        print("Goal has not been achieved in  = ", self.not_goal_count)            
        self.execution_count += 1
        print(f"Execution count: ",self.execution_count) #Printing point coding
        print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

        
        if self.gazebo:
        	self.reset_simulation()
        self.step_cnt = 0  # Resetea el contador de pasos
        self.last_distance = 0  # Resetea la última distancia al objetivo
        self.real_robot_initial_pose_set = False  # Reset initial pose flag

        dummy_action = self.action_size // 2
        state = self.take_observation(dummy_action)
        return np.asarray(state)

            
    
    
       
    def render(self, mode=None,  close=False):
        pass

    def close(self):
        print ("fake close")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

