from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNN(nn.Module):
    """Dilated CNN."""

    def __init__(self,
                 img_size: int,
                 hidden_channels: List[int],
                 max_num_jumps: int):
        """Initializes an object of DCNN.

        Args:
            img_size: Input image size.
            hidden_channels: The list of hidden channels.
            max_num_jumps: The maximum number of jumps.
        """
        # TODO: add dropout
        super().__init__()
        self.max_num_jumps = max_num_jumps

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=hidden_channels[0],
                               kernel_size=5,
                               padding=2,
                               bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])

        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0],
                               out_channels=hidden_channels[1],
                               kernel_size=3,
                               dilation=2,
                               padding=2,
                               bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])

        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1],
                               out_channels=hidden_channels[2],
                               kernel_size=3,
                               dilation=3,
                               padding=3,
                               bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((img_size // 8)**2 * hidden_channels[2], 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, max_num_jumps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts inflection points.

        Args:
            x: A batch of cross-similarity matrices of shape (B, 1, H, W),
              where `B` is the batch size, `H` and `W` are the height and
              width, respectively.

        Returns:
            A batch of inflection points of shape (B, max_num_jumps, 2).
        """
        # Pre-normalize
        x = x * 2 - 1

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        #x = F.relu(x)

        bs = x.shape[0]
        x = x.reshape((bs, self.max_num_jumps, 2))
        return x
