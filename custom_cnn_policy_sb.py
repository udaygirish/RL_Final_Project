import numpy as np
import gym
from gym import spaces
from IMG_POS_CNN import CombinedFeatureNetwork
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import A2C
import torch
import torch.nn as nn
import numpy as np

import gym
import torch.nn as nn
    
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 9):
        super().__init__(observation_space, features_dim)

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=64, stride=1),  
            nn.ReLU()
        )

        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=32, stride=1),
            nn.ReLU()
        )

        self.depth_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=16, stride=1),
            nn.ReLU()
        )

        self.depth_conv4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=8, stride=1),
            nn.ReLU()
        )

        self.depth_conv5 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=4, stride=1),
            nn.ReLU()
        )

        self.depth_fc1 = nn.Sequential(
            nn.Linear(5000, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.pos_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.depth_pos_combine_fc = nn.Sequential(
            nn.Linear(16+16, 16),  
            nn.ReLU(),
            nn.Linear(16, 9)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        pos = observations['pos']
        depth = observations['depth']

        print("Input depth: ", depth.size())
        print("Input pos: ", pos.size())
        
        depth_features = self.depth_conv1(depth)
        print("Output of depth_conv1: ", depth_features.size())
        
        depth_features = self.depth_conv2(depth_features)
        print("Output of depth_conv2: ", depth_features.size())
        
        depth_features = self.depth_conv3(depth_features)
        print("Output of depth_conv3: ", depth_features.size())

        depth_features = self.depth_conv4(depth_features)
        print("Output of depth_conv4: ", depth_features.size())

        depth_features = self.depth_conv5(depth_features)
        print("Output of depth_conv5: ", depth_features.size())
        
        depth_features = depth_features = depth_features.view(depth_features.size(0), -1)
        print("Output of depth_serial: ", depth_features.size())
        
        depth_features = self.depth_fc1(depth_features)
        print("Output of depth_fc1: ", depth_features.size())

        # Process relative XYZ position with FCN
        position_features = self.pos_fc(pos)
        print("Output of pos_fc: ", position_features.size())

        # Combine features
        combined_features = torch.cat([depth_features, position_features], dim=1)
        print("Combined features: ", combined_features.size())

        # Process combined features with FCN
        final = self.depth_pos_combine_fc(combined_features)
        print("Final output: ", final.size())

        print(" ")

        return final

class CustomCNNPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)  # Only pass positional arguments
        self.features_extractor = CustomFeatureExtractor(self.observation_space)
        self.iteration = 0

    def _predict(self, observation: torch.Tensor | torch.Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return super()._predict(observation, deterministic)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.iteration == 0:  # if it's the second iteration (0-indexed)
            print(observations)
        self.iteration += 1  # increment the counter

        return self.features_extractor(observations)


policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=9),
)

# Define a dummy environment for testing
class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            'pos': gym.spaces.Box(low=-10, high=10, shape=(3,)),  # XYZ position
            'depth': gym.spaces.Box(low=0, high=255, shape=(144, 144))  # Depth image
        })
        self.action_space = gym.spaces.Discrete(9)  # Define the action space

        # Initialize state with random values (optional)
        self.state = {
            'pos': torch.randn(1, 3),
            'depth': torch.randn(1, 1, 144, 144)
        }

    def reset(self):
        # Generate random observations
        pos = torch.randn(1, 3)
        depth = torch.randn(1, 1, 144, 144)

        # Create the observation dictionary and return it
        observation = {
            'pos': pos,
            'depth': depth
        }
        return observation

    def step(self, action):
        # Use the current state from the previous reset
        return self.state.copy(), 0, False, {}
    
    def render(self, mode='human'):
        pass  # No rendering for this dummy environment

    def close(self):
        pass  # No cleanup necessary for this dummy environment

    def seed(self, seed=None):
        # Optionally use the seed to control the randomness of your environment
        # (e.g., setting a random seed for numpy.random)
        pass


# Create an instance of the dummy environment
denv = DummyEnv()

obs, _ = denv.reset()
denv.render()

print(denv.observation_space)
print(denv.action_space)
print(denv.action_space.sample())

# Vectorize the environment
vec_env = make_vec_env(lambda: DummyEnv(), n_envs=1, env_kwargs=dict() )

# Create the model with the vectorized environment and the custom policy
model = A2C(policy=CustomCNNPolicy, env=vec_env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(10)

