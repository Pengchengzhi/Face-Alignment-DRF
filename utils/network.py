import torch.nn as nn
from collections import OrderedDict

class CNN_module(nn.Module):
  def __init__(self):
    super(CNN_module, self).__init__()

    self.conv = nn.Sequential(
      OrderedDict([('conv1', nn.Conv2d(in_channels=3,out_channels=5,kernel_size=3,stride=1,padding=1)),
             ('relu1', nn.ReLU()),
             ('pool1', nn.MaxPool2d(kernel_size=2)), #Input(384,384,3), Output(192,192,5)

             ('conv2', nn.Conv2d(in_channels=5,out_channels=10,kernel_size=3,stride=1,padding=1)),
             ('relu2', nn.ReLU()),
             ('pool2', nn.MaxPool2d(kernel_size=2)), #Input(192,192,5), Output(96,96,10)

             ('conv3-1', nn.Conv2d(in_channels=10,out_channels=13,kernel_size=3,stride=1,padding=1)),
             ('relu3-1', nn.ReLU()),
             ('conv3-2', nn.Conv2d(in_channels=13,out_channels=16,kernel_size=3,stride=1,padding=1)),
             ('relu3-2', nn.ReLU()),
             ('pool3', nn.MaxPool2d(kernel_size=2)), #Input(96,96,10), Output(48,18,16)
             
             ('conv4-1', nn.Conv2d(in_channels=16,out_channels=19,kernel_size=3,stride=1,padding=1)),
             ('relu4-1', nn.ReLU()),
             ('conv4-2', nn.Conv2d(in_channels=19,out_channels=22,kernel_size=3,stride=1,padding=1)),
             ('relu4-2', nn.ReLU()),
             ('pool4', nn.MaxPool2d(kernel_size=2)), #Input(48,18,16), Output(24,24,22)

             ('conv5-1', nn.Conv2d(in_channels=22,out_channels=25,kernel_size=3,stride=1,padding=1)),
             ('relu5-1', nn.ReLU()),
             ('conv5-2', nn.Conv2d(in_channels=25,out_channels=28,kernel_size=3,stride=1,padding=1)),
             ('relu5-2', nn.ReLU()),
             ('pool5', nn.MaxPool2d(kernel_size=2)),])) #Input(24,24,22), Output(12,12,28)

            # should be (28,7,7) here
            #  ('conv6-1', nn.Conv2d(in_channels=28,out_channels=31,kernel_size=3,stride=1,padding=1)),
            #  ('relu6-1', nn.ReLU()),
            #  ('conv6-2', nn.Conv2d(in_channels=31,out_channels=34,kernel_size=3,stride=1,padding=1)),
            #  ('relu6-2', nn.ReLU()),
            #  ('pool6', nn.MaxPool2d(kernel_size=2)),])) #Input(12,12,28), Output(6,6,34)
      
    self.fc = nn.Sequential(
      OrderedDict([('fc1', nn.Linear(in_features=28*7*7, out_features=128)),
            ('fc2', nn.Linear(in_features=128, out_features=64)),
            ('fc3', nn.Linear(in_features=64, out_features=8)),]))

  def forward(self, x):
    out = self.conv(x)
    #print("Shape after conv",out.shape)
    out = out.view(-1, 28*7*7)
    #print("Shape flattened",out.shape)
    out = self.fc(out)
    return out







