import numpy as np
import math
from scipy.spatial import KDTree

def find_closest_point_kdtree(current_location, points, goal, radius):
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

def compute_reward(startState, goalState, currentState):    
    curr = np.array([currentState["position"].x_val, currentState["position"].y_val, currentState["position"].z_val]) 
    
    prevState = np.array([currentState["prev_position"].x_val, currentState["prev_position"].y_val, currentState["prev_position"].z_val])
    
    # Penalize collisions
    if currentState["collision"]:
        reward = -100
    else:
        ####### Distance Based Reward #######
        # Calculate distance to goal state - distance only using x
        dist_to_goal_x = np.linalg.norm(curr[0] - goalState[0])
        dist_to_goal_y = np.linalg.norm(curr[1] - goalState[1])
        dist_to_goal_z = np.linalg.norm(curr[2] - goalState[2])
        
        # Reward for progress towards the goal
        reward_dist = math.exp(-0.1 * dist_to_goal_x) + 0.75*math.exp(-0.1 * dist_to_goal_y) + 0.5*math.exp(-0.1 * dist_to_goal_z) 
        #####################################

        ####### Angle Based Reward ##########
        # Compute angle of subtended at goal between goal-start and goal-current
        #print(type(startState), type(goalState), type(curr), type(prevState))
        
        angle = np.arccos(np.dot((goalState - startState), (curr - goalState)) / (np.linalg.norm(goalState - startState) * np.linalg.norm(curr - goalState)))
        # Give more reward if the angle is less then 45 degrees and give penalty if the angle is greater than 45 degrees
        if angle < np.pi/4:
            reward_angle = 1
        else:
            reward_angle = 1 - (2*angle / np.pi)
        #####################################

        ########### Speed Reward ############        
        reward_speed = np.linalg.norm([currentState["velocity"].x_val, currentState["velocity"].y_val, currentState["velocity"].z_val])
        #####################################

        ########### KDTree Reward ###########
        num_points = 100
        t = np.linspace(0, 1, num_points)
        points = np.outer(1 - t, startState) + np.outer(t, goalState)
        
        # Find the closest point on start-goal line which is within a radius to the quadrotor using a KDTree
        closest_points = find_closest_point_kdtree(curr, points, goalState, 10)
        
        # Check distance of first element in closest_points to goal give reward based on distance
        reward_kdtree = 0
        if closest_points is not None:
            dist_to_goal_closest = np.linalg.norm(closest_points[0] - goalState)
            reward_kdtree = math.exp(-0.1 * dist_to_goal_closest)
        #####################################

        ######### Smoothness Reward #########
        smoothness_penalty = np.linalg.norm(curr - prevState)
        #####################################

        ############ Total Reward ###########        
        reward = 5 * reward_dist + reward_angle + reward_kdtree + reward_speed - 0.01 * smoothness_penalty
        #####################################

    done = 0
    # Terminate if the quadrotor is within a certain radius of the goal
    dist_to_goal = np.linalg.norm(curr - goalState)
    if dist_to_goal < 5 or reward <= -10:
        done = 1

    return reward, done