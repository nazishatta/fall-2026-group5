import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 1 input channel, 6 output channels, 5x5 convolution
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # pad to maintain 28x28 before pooling or use 32x32. LeNet originally expects 32x32. For 28x28 we can use padding=2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Penultimate feature layer info
        self.feature_layer_name = 'fc2'
        self.feature_dim = 84
        
        # Linear layers
        # With padding=2 on conv1, output is 28x28. Pool1 -> 14x14. Conv2(5x5, no pad) -> 10x10. Pool2 -> 5x5.
        # 16 channels * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_feature_layer_name(self):
        return self.feature_layer_name
