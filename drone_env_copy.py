import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
from scipy.spatial import KDTree
import sys 
BASE_PATH = "/home/udaygirish/Projects/WPI/Unreal_Stuff/AirSim/PythonClient/reinforcement_learning/"
sys.path.append(BASE_PATH + "airgym/envs")
sys.path.append(BASE_PATH + "airgym/")

from drone_reward import *

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip = ip_address)
        self.action_space = spaces.Discrete(9)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        # Define the start and goal states
        self.startState = np.array([0, 0, 0])
        self.goalState = np.array([110, -15, -12]) #([110, -50, 10])

        # # Define the Points on the path
        # distance = np.linalg.norm(self.goalState - self.startState)

        # Calculate the number of points to sample
        self.num_points = 100

        # Create a vector of size num_points ranging from 0 to 1
        t = np.linspace(0, 1, self.num_points)

        # Sample points along the line
        points = np.outer(1 - t, self.startState) + np.outer(t, self.goalState)

        # Transpose the points array to get a list of points
        self.pts = points
        
        print("startState: ", self.startState)
        print("goalState: ", self.goalState)
        # print("Points Shape: ", self.pts)
        
    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()                    
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        
        # Set home position and velocity   
        
        # Left Env Start Goal  
        #print("Im in Move to position - Start")
        self.drone.moveToPositionAsync(0, 0, 5, 2).join()
        #print("Im in Move to position - Velocity")
        self.drone.moveByVelocityAsync(0, 0, 0, 2).join()
        self.startState = np.array([0, 0, 5])

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))
        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state. kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
 
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        temp_curr_position = np.array([self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val])
        relative_position = temp_curr_position - self.startState
        relative_position = relative_position.reshape(1, 3)
        return image
    
    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            7,
        ).join()

    def calculate_rewards(self):
        reward, done = compute_reward(self.startState, self.goalState, self.state)
        return reward, done

    def _compute_reward(self):          
        reward, done = self.calculate_rewards()        
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        
        print("")
        print("End state", self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val)
        print("Distance to goal", np.linalg.norm(np.array([self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val]) - self.goalState))
        print("Reward: ", reward)
        print("")
        
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()   
        #self.startState = self.state['position']
        return self._get_obs()

    def interpret_action(self, action):
        print("Action", action)           
        if action == 0:
            quad_offset = (self.step_length, 0, 0)  # Move forward
        elif action == 1:
            quad_offset = (0, self.step_length, 0) # Move right
        elif action == 2:
            quad_offset = (0, 0, self.step_length) # Move up
        elif action == 3:
            quad_offset = (0, -self.step_length, 0) # Move left
        elif action == 4:
            quad_offset = (0, 0, -self.step_length) # Move down
        elif action == 5:
            quad_offset = (self.step_length, self.step_length, 0) # Move forward and right
        elif action == 6:
            quad_offset = (self.step_length, -self.step_length, 0) # Move forward and left
        elif action == 7:
            quad_offset = (self.step_length, 0, self.step_length) # Move forward and up
        elif action == 8:
            quad_offset = (self.step_length, 0, -self.step_length) # Move forward and down
        
        return quad_offset
