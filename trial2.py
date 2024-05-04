import os

import torch
import torch.nn as nn

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
from sinabs.from_torch import from_model



############# MODEL ###############
import torch
import torch.nn as nn
from typing import List
import sinabs
import sinabs.layers as sl

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
                stride=2
            else:
                in_channels = channels
                stride=1

            conv.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride))
            conv.append(nn.BatchNorm2d(channels))
            conv.append(sl.IAFSqueeze(*args, **kwargs))
            if i != 0:
              conv.append(sl.SumPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels * 4 * 4, 512),
            sl.IAFSqueeze(*args, **kwargs),

            nn.Dropout(0.5),
            nn.Linear(512, 110),
            sl.IAFSqueeze(*args, **kwargs),
            nn.Linear(110,11),
            #sl.SumPool2d((10,1), stride=(10,1)),
            sl.IAFSqueeze(*args, **kwargs),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def return_sequential(self):
      return self.conv_fc

net = DVSGestureNet(batch_size=1, channels=8)
cpu_snn = net.return_sequential().to(device="cpu")
hardware_compatible_model = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 128, 128), discretize=True, dvs_input=False)

hardware_compatible_model.to(
    device="speck2fmodule",  # TODO or speck2edevkit:0
    monitor_layers=["dvs", -1],  # Last layer
    chip_layers_ordering="auto",
)

icons_folder_path = str(os.path.abspath(__file__)).split("/")[:-1]
icons_folder_path = os.path.join("/", os.path.join(*icons_folder_path), "icons")

visualizer = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    add_readout_plot=True,
    spike_collection_interval=500,
    readout_images=sorted(
        [os.path.join(icons_folder_path, f) for f in os.listdir(icons_folder_path)]
    ),
)
visualizer.connect(hardware_compatible_model)