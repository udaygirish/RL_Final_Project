
from tracemalloc import start
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
from torch.nn.modules.linear import Linear

import torchvision.models as pre_models
import numpy as np
import torch.nn.functional as F
from gym import spaces

'''
New Model for improving the current Depth only model
Adding Image  + Feature Vector (Relative Motion) and Training a combination'''

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "position":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
    
    

# class Custom_CNN_FC(BaseFeaturesExtractor):
#     '''
#     param observation_space: (gym.Space)
#     param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
    
#     '''

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 84, state_feature_dim=9):
#         super(Custom_CNN_FC, self).__init__(observation_space, features_dim)
#         # We assume HxWxC images (channels first)

#         #assert state_feature_dim > 0
#         self.feature_num_state = state_feature_dim
#         self.feature_all = None

#         # Input Size = 84*84
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # [1, 8, 42, 42]

#             nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # [1, 16, 21, 21]

#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # [1, 32, 10, 10]

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # [1, 64, 5, 5]

#             nn.Flatten()
#         )

#         self.linear_encoder = nn.Sequential(
#             # Input 3 
#             nn.Linear(3, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#         )

#         # Compute shape by doing one forward pass
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         print(observations.shape)
#         depth_img = observations['image']
#         state_feature = observations['state']
#         # Shuffling the image to (1, 84, 84)
#         depth_img = depth_img.permute(2, 0, 1)

#         # Shuffling the state feature to (1, 3)

#         state_feature = state_feature.permute(1, 0)

#         cnn_out = self.cnn(depth_img.unsqueeze(0))
#         linear_out = self.linear_encoder(state_feature)

#         x = th.cat((cnn_out, linear_out), dim=1)

#         self.feature_all = x

#         return x
    

