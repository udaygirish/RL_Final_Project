import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
from scipy.spatial import KDTree

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
        # start - 10, 0, 10
        # goal - 200, 0, 10
        #print("Im in Move to position - Start")
        self.drone.moveToPositionAsync(0, 0, 5, 2).join()
        #print("Im in Move to position - Velocity")
        self.drone.moveByVelocityAsync(0, 0, 0, 2).join()

        # Middle Env Start Goal  
        # Start - 0, 50, 10
        # Goal - 200, 50, 10
        # self.drone.moveToPositionAsync(0, 0, 5, 1).join()
        # self.drone.moveByVelocityAsync(0, 0, 0, 2).join()

        # Right Env Start Goal
        # Start - 0, 50, 10
        # Goal - 200, 50, 10  
        #print("Im in Move to position - End")
        #self.drone.moveToPositionAsync(110, -15, -20, 5).join()
        #self.drone.moveByVelocityAsync(0, 0, 0, 2).join()


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
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

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

    def find_closest_point_kdtree(self, current_location, points, goal, radius):
        # Ensure the inputs are numpy arrays
        current_location = np.array(current_location,dtype=np.int8)
        points = np.array(points, dtype=np.int8)
        goal = np.array(goal, dtype=np.int8)

        points = points.reshape(-1,3)

        # Create a KDTree from the points
        tree = KDTree(points)
        
        # Find the indices of the points within the radius
        indices = tree.query_ball_point(current_location, radius)

        # If no points are within the radius, return None
        if len(indices) == 0:
            return None

        # Get the points within the radius
        points_within_radius = points[indices]

        # Calculate the Euclidean distance from the goal to each point within the radius
        distances_to_goal = np.linalg.norm(points_within_radius - goal, axis=1)

        # Get the indices that would sort the distances
        sorted_indices = np.argsort(distances_to_goal)

        # Sort the points within the radius by their distance to the goal
        points_within_radius_sorted = points_within_radius[sorted_indices]

        return points_within_radius_sorted

    def calculate_rewards(self, current_state, points_within_radius_sorted):
        # Calculate the Euclidean distance from the current state to each point
        dist_point = np.linalg.norm(points_within_radius_sorted - current_state, axis=1)

        # Create a reward array that is inversely proportional to the index
        rewards = 5 / (np.arange(len(dist_point)) + 1)

        # Get the index of the minimum value in dist_point
        min_index = np.argmin(dist_point)

        # If the index of the minimum value is within the first 5 indices, assign a high reward
        if min_index < 20:
            reward = 200
        else:
            reward = 0 

        return reward

    def _compute_reward(self):   
        # quad_pt = np.array(list((self.state["position"].x_val, self.state["position"].y_val,self.state["position"].z_val,)))

        # thresh_dist = 10
        # beta = 1
        # search_radius = 100
                
        # pts = self.pts
        
        # # if self.state["collision"]:
        # #     reward = -100
        # # else:
        # #     dist = 10000000
        # #     for i in range(0, len(pts) - 1):
        # #         dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(pts[i] - pts[i + 1]))

        # #     if dist > thresh_dist:
        # #         reward = -10
        # #     else:
        # #         reward_dist = math.exp(-beta * dist) - 0.5
        # #         reward_speed = (np.linalg.norm([self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val,])- 0.5)
        # #         reward = reward_dist + reward_speed
        
        # if self.state["collision"]:
        #     reward = -100
        # else:
        #     # reward_point = self.calculate_rewards(quad_pt, closest_points)
            
        #     dist = np.linalg.norm(quad_pt - self.goalState) - 0.5
        #     reward_dist = math.exp(beta * (1.0 / (dist + 1e-6)))
            
        #     reward_speed = ( np.linalg.norm( [self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val,]))                        
            
        #     reward = 90*reward_dist + 10*reward_speed
            
        #     print("reward dist: ", reward_dist, "Reward speed: ", reward_speed, "Reward: ", reward)

        # done = 0
        # if reward <= -10:
        #     done = 1
            
        # return reward, done
        # return 0,0
        quad_pt = np.array([self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val])

        # Penalize collisions
        if self.state["collision"]:
            reward = -100
        else:
            # Calculate distance to goal
            dist_to_goal = np.linalg.norm(quad_pt - self.goalState)

            # Reward for progress towards the goal
            reward_dist = math.exp(-0.1 * dist_to_goal)

            # Speed reward
            reward_speed = np.linalg.norm([self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val])

            # Smoothness reward
            prev_pos = np.array([self.state["prev_position"].x_val, self.state["prev_position"].y_val, self.state["prev_position"].z_val])
            smoothness_penalty = np.linalg.norm(quad_pt - prev_pos)

            # Total reward
            reward = 5* reward_dist + 0.5 * reward_speed - 0.01 * smoothness_penalty

        done = 0
        if reward <= -10:
            done = 1
            
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
        print("")
        self.startState = self.state["position"]
        print("")
        return self._get_obs()

    # def interpret_action(self, action):
    #     print("Action", action)
    #     if action == 0:
    #         quad_offset = (self.step_length, 0, 0)
    #     elif action == 1:
    #         quad_offset = (0, self.step_length, 0)
    #     elif action == 2:
    #         quad_offset = (0, 0, self.step_length)
    #     elif action == 3:
    #         quad_offset = (-self.step_length, 0, 0)
    #     elif action == 4:
    #         quad_offset = (0, -self.step_length, 0)
    #     elif action == 5:
    #         quad_offset = (0, 0, -self.step_length)
    #     elif action == 6:
    #         quad_offset = (0, 0, 0)
    #     elif action == 7:
    #         quad_offset = (self.step_length, self.step_length, 0)
    #     elif action == 8:
    #         quad_offset = (-self.step_length, self.step_length, 0)
    #     elif action == 9:
    #         quad_offset = (self.step_length, -self.step_length, 0)
    #     elif action == 10:
    #         quad_offset = (-self.step_length, -self.step_length, 0)
    #     elif action == 11:
    #         quad_offset = (self.step_length, 0, self.step_length)
    #     elif action == 12:
    #         quad_offset = (-self.step_length, 0, self.step_length)
    #     elif action == 13:
    #         quad_offset = (self.step_length, 0, -self.step_length)
    #     elif action == 14:
    #         quad_offset = (-self.step_length, 0, -self.step_length)

    #     return quad_offset

    def interpret_action(self, action):
        print("Action", action)
        if action == 0:
            quad_offset = (self.step_length, 0, 0)  # Move forward
        elif action == 1:
            quad_offset = (0, self.step_length, 0) # Move right
        elif action == 2:
            quad_offset = (0, 0, self.step_length) # Move up
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0) # Move backward
        elif action == 4:
            quad_offset = (0, -self.step_length, 0) # Move left
        elif action == 5:
            quad_offset = (0, 0, -self.step_length) # Move down
        elif action == 6:
            quad_offset = (self.step_length, self.step_length, 0) # Move forward and right
        elif action == 7:
            quad_offset = (self.step_length, -self.step_length, 0) # Move forward and left
        elif action == 8:
            quad_offset = (self.step_length, 0, self.step_length) # Move forward and up
        elif action == 9:
            quad_offset = (self.step_length, 0, -self.step_length) # Move forward and down

        return quad_offset
