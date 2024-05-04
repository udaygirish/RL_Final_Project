import torch
import torch.nn as nn
from torchsummary import summary

class CustomCombinedExtractor(nn.Module):
    def __init__(self):
        super(CustomCombinedExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 21 * 21, 64),
            nn.ReLU()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU()
        )

    def forward(self, images, positions):
        conv_output = self.conv_layers(images)
        linear_output = self.linear_layers(positions)
        combined_output = torch.cat((conv_output, linear_output), dim=1)
        return combined_output

# Assuming input shapes (32, 1, 84, 84) for images and (32, 3) for positions
model = CustomCombinedExtractor()

# Specify the file path
output_file = "model_summary.txt"

# Open the file in write mode and redirect the output
with open(output_file, "w") as f:
    summary_str = summary(model, [(1, 84, 84), (3,)])
    f.write(summary_str)

print(f"Model summary saved to {output_file}")
