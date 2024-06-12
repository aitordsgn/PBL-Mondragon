###############################################################################
#                                                                             #
# Project: TurtleBot3 Reinforcement Learning                                  #
# Author: Aitor Rey Ortega                                                    #
# University: Mondragon Unibertsitatea                                        #
#                                                                             #
# Description:                                                                #
# This script implements the TurtleBot3 environment for training a            #
# reinforcement learning agent to navigate and reach a goal within a          #
# 4x4m area. The code uses ROS and Gazebo for simulation.                     #
#                                                                             #
# File: turtlebot3_ros1_dis_PBL_env.py                                        #
#                                                                             #
###############################################################################

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy
from numpy import array
import numpy as np

import math

import os
import signal
import subprocess
import time
import copy

from gazebo_msgs.srv import SpawnModel, DeleteModel
#from gazebo_msgs.srv import DeleteEntity
#from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan

import rospy
import rosnode
from std_srvs.srv import Empty
#from rclpy.node import Node
#from rclpy.qos import QoSProfile
#from rclpy.qos import qos_profile_sensor_data
#from std_srvs.srv import Empty

from os import path
import random

class Turtlebot3Ros1DisEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
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
        


        #self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'
        self.goal_displayed = False

        self.last_distance = 0.0
        self.step_cnt = 0
        self.max_step_episode = 200

        self.allowed_min_obstacle_distance = 0.20
        self.goal_distance_acept = 0.25
        self.goal_distance = 0
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        """************************************************************
        ** Environment configuration variables
        ************************************************************"""
        self.real_robot_initial_pose_set = False
        self.real_robot_initial_pose = [0,0]

        # class variables
        self._observation_msg = None
        self.scan_msg_ready = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""

        # Initialise subscribers
        print ("1 ")
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
        print ("2 ")
        
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
        print ("3")
        self.stage = 1
        if self.gazebo:
            self.goal_pose_x = 0.0
            self.goal_pose_y = 0.0
            self.generate_goal_pose()
        else:
            initial_x, initial_y = self.real_robot_initial_pose
            self.goal_pose_x = initial_x + 1.3
            self.goal_pose_y = initial_y - 1.6
            self.generate_goal_pose()
        self.maxt_distance = 3.0
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
         
        #Check if it is not gazebo
        if not self.gazebo and not self.real_robot_initial_pose_set:
            self.real_robot_initial_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            self.real_robot_initial_pose_set = True

        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

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
        filtered_scan_ranges = [value for value in self.scan_ranges if value != 0.0]

        self.min_obstacle_distance = min(filtered_scan_ranges)
        self.min_obstacle_angle = numpy.argmin(filtered_scan_ranges)
        self.scan_msg_ready = True

    def reset_simulation(self):
        if self.gazebo:        
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
        self.execution_count += 1
        print(f"Execution count: ",self.execution_count) #Printing point coding
    def take_observation(self, action):
        """
        Take observation from the environment and return it.
        TODO: define return type
        """
        # Take an observation

        self.scan_msg_ready = False

        a = 0
        while self.scan_msg_ready is False:
            a += 1 

        observation = copy.deepcopy(self.scan_ranges)
        goal_distance = copy.deepcopy(self.goal_distance)
        goal_angle = copy.deepcopy(self.goal_angle)
        scan_range = []
        #print('observation', len(observation))
        if self.gazebo:
            # Limit the maximun distance to 3.5 meters and then normalize
            for i in range(0,len(observation)):
                if observation[i] == float('Inf'):
                	scan_range.append(3.5/3.5)
                elif np.isnan(observation [i]):
                	scan_range.append(0)
                elif observation[i] > 3.5:
                	scan_range.append(3.5/3.5)
                else:
                	scan_range.append(observation[i]/3.5)
        else:
            # Limit the maximun distance to 3.5 meters and then normalize
            for i in range(0,len(observation),20):
                if observation[i] == float('Inf'):
                	scan_range.append(3.5/3.5)
                elif np.isnan(observation [i]):
                	scan_range.append(0)
                elif observation[i] > 3.5:
                	scan_range.append(3.5/3.5)
                else:
                	scan_range.append(observation[i]/3.5)
        observation = scan_range + [(goal_angle+math.pi)/(2*math.pi), goal_distance/10.0, action]
        #print('observation_len', len(observation))
        #print('observation_array', observation)
        return observation

    def setReward(self, state):
        reward = 0
        done = False        
        self.step_cnt = self.step_cnt + 1
        
        if self.last_distance == 0:
            self.last_distance = self.goal_distance
        distance_rate = (self.last_distance - self.goal_distance) / self.last_distance
        self.last_distance = self.goal_distance
        
        if distance_rate >= 0:
            reward = 1
        else:
            reward = -2
        # self.goal_angle

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0

        done = False
        if self.min_obstacle_distance < self.allowed_min_obstacle_distance:
            print("Collision!!")
            reward = -200
            self.cmd_vel_pub.publish(vel_cmd)
            done = True

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
            reward = 200
            done = True            
        
        if self.step_cnt > self.max_step_episode:
            self.step_cnt = 0
            print("Max step achieved!!")
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
            done = True

        return reward, done

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - reward
            - done (status)
            - action
            - observation
            - dictionary (#TODO clarify)
        """

        if self.continuous == True:
            ang_vel = action[0]*1.0
        else:            
            ang_vel = ((self.action_size - 1)/2 - action) * self.max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.cmd_vel_pub.publish(vel_cmd)
        self.scan_msg_ready = False

        state = self.take_observation(action)
        reward, done = self.setReward(state)
        observation = np.asarray(state)
        truncated = 0
        return observation, reward, done, truncated, {}
        
    def generate_goal_pose(self):
        goal_ok = False
        if self.gazebo:
            if self.stage != 4:
                while goal_ok == False:
                    self.goal_pose_x = random.randrange(-15, 16) / 10.0
                    self.goal_pose_y = random.randrange(-15, 16) / 10.0
                    if (self.goal_pose_x > -2.0 and self.goal_pose_x < -1.45) or (self.goal_pose_x > -0.55 and self.goal_pose_x < 0.55) or (self.goal_pose_x > 1.45 and self.goal_pose_x < 2.0):
                        goal_ok = True
                    if (self.goal_pose_y > -2.0 and self.goal_pose_y < -1.45) or (self.goal_pose_y > -0.55 and self.goal_pose_y < 0.55) or (self.goal_pose_y > 1.45 and self.goal_pose_y < 2.0):
                        goal_ok = True

            else:
                goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                                [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0]]
                index = random.randrange(0, 10)
                self.goal_pose_x = goal_pose_list[index][0]
                self.goal_pose_y = goal_pose_list[index][1]        
        else:
            initial_x, initial_y = self.real_robot_initial_pose
            self.goal_pose_x = initial_x + 1.3
            self.goal_pose_y = initial_y + 1.4
        print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

    def spawn_entity(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        target_future = self.spawn_entity_client.call_async(req)
        #rclpy.spin_until_future_complete(self.node, target_future)

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        print ("Try to delete entity")
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        target_future = self.delete_entity_client.call_async(req)
        #rclpy.spin_until_future_complete(self.node, target_future)

    def reset(self):  
        self.reset_simulation()
        action = self.action_size /2
        state = self.take_observation(action)
        return np.asarray(state)
        
    def render(self, mode=None,  close=False):
        pass

    def close(self):
        print ("fake close")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        

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


