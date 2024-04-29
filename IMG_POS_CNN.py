import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CombinedFeatureNetwork(nn.Module):
    def __init__(self):

        super(CombinedFeatureNetwork, self).__init__()

        self.depth_conv1 = nn.Sequential(
            # Assuming depth image has 1 channel
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


    def forward(self, pos, depth):
        # Process image with convolutional layers

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

        return final

    
# # Assuming you have a single image of size 144x144
image = torch.randn(1, 1, 144, 144)  # Depth images usually have 1 channel

# And a single position
position = torch.randn(1, 3)  # Reshape to have a batch size of 1

# Create the model
model = CombinedFeatureNetwork()

# Run the model
output = model(position, image)
