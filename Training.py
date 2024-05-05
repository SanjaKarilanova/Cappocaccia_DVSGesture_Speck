#### LOAD DATA ####

import tonic
from tonic.transforms import ToFrame
from tonic.datasets import nmnist
from tonic import transforms
import os
import numpy as np

root = "/"

transform = transforms.Compose([
    transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins=16, include_incomplete=True),
    lambda x: x.astype(np.float32),
])

testset = tonic.datasets.DVSGesture(save_to="data/", train=False, transform=transform)
trainset = tonic.datasets.DVSGesture(save_to="data/", train=True, transform=transform)

events, label = trainset[0]
events[0].shape

#### DEFINE MODEL ###

import torch
import torch.nn as nn
from typing import List
import sinabs
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



from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

epochs = 20
lr = 1e-3
batch_size = 40
num_workers = 4
n_time_steps=16
device = "cuda:0" if torch.cuda.is_available() else "cpu"
shuffle = True

snn_train_dataloader = DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True) #  num_workers=num_workers,
snn_test_dataloader = DataLoader(testset, batch_size=batch_size, drop_last=True, shuffle=False) #  num_workers=num_workers,


net = DVSGestureNet(batch_size=1)
net = net.to(device=device)

optimizer = Adam(params=net.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):

    # train
    train_p_bar = tqdm(snn_train_dataloader)
    for data, label in train_p_bar:#snn_train_dataloader:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        optimizer.zero_grad()
        output = net(data)
        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        loss = criterion(output, label)
        #print(loss.device)
        # backward
        loss.backward()
        optimizer.step()

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in net.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        # set progressing bar
        train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

    # validate
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(snn_test_dataloader)
        for data, label in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            output = net(data)
            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            # compute the total correct predictions
            correct_predictions.append(pred.eq(label.view_as(pred)))
            # set progressing bar
            test_p_bar.set_description(f"Epoch {e} - BPTT Testing Model...")

        correct_predictions = torch.cat(correct_predictions)
        print(f"Epoch {e} - BPTT accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%")

    torch.save(net.state_dict(), "model.pth")