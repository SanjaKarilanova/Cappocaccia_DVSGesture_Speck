import torch
import torch.nn as nn
import sinabs.layers as sl

class DVSGestureNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = []
        # dimensions at input of IF layer
        # 64x64, 64x64, 32x32
        channels = [2, 16, 64, 128, 64, 8]  # 15 is the most we can do for the first conv layer
        kernel_size = [2, 2, 2, 2, 2]
        stride = [2, 2, 2, 2, 2]

        for i in range(5):
            conv.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size[i], stride=stride[i]))
            conv.append(nn.BatchNorm2d(channels[i + 1]))
            conv.append(sl.IAFSqueeze(*args, **kwargs))
            # if i != 0:
            #  conv.append(sl.SumPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels[-1] * 4 * 4, 512),
            sl.IAFSqueeze(*args, **kwargs),

            nn.Dropout(0.5),
            nn.Linear(512, 110),
            sl.IAFSqueeze(*args, **kwargs),
            nn.Linear(110, 11),
            # sl.SumPool2d((10,1), stride=(10,1)),
            sl.IAFSqueeze(*args, **kwargs),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def return_sequential(self):
        return self.conv_fc
