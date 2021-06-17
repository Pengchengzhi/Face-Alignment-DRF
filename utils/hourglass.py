# Thanks Raymon, 
# for the implementation of Hourglass
# https://github.com/raymon-tian/hourglass-facekeypoints-detection

import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.autograd import Variable



class Residual(nn.Module):
  def __init__(self,ins,outs):
    super(Residual,self).__init__()
    self.convBlock = nn.Sequential(
        nn.BatchNorm2d(ins),
        nn.ReLU(inplace=True),
        nn.Conv2d(ins,int(outs/2),1),
        nn.BatchNorm2d(int(outs/2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(outs/2),int(outs/2),3,1,1),
        nn.BatchNorm2d(int(outs/2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(outs/2),outs,1)
    )
    if ins != outs:
      self.skipConv = nn.Conv2d(ins,outs,1)
    self.ins = ins
    self.outs = outs
  def forward(self,x):
    residual = x
    x = self.convBlock(x)
    if self.ins != self.outs:
      residual = self.skipConv(residual)
    x += residual
    return x



class Lin(nn.Module):
  def __init__(self,numIn=128,numout=68):
    super(Lin,self).__init__()
    self.conv = nn.Conv2d(numIn,numout,1)
    self.bn = nn.BatchNorm2d(numout)
    self.relu = nn.ReLU(inplace=True)
  def forward(self,x):
    return self.relu(self.bn(self.conv(x)))


class Tune(nn.Module):
  def __init__(self):#,numIn=128,numout=68):
    super(Tune,self).__init__()
    # self.conv = nn.Conv2d(numIn,numout,1)
    # self.bn = nn.BatchNorm2d(numout)
    # self.relu = nn.ReLU(inplace=True)
  def forward(self,x):
    return x #self.relu(self.bn(self.conv(x)))



class HourGlass(nn.Module):
  def __init__(self,n=4,f=128):
    """
    :param n: hourglass num of layers
    :param f: hourglass num of feature maps
    :return:
    """
    super(HourGlass,self).__init__()
    self._n = n
    self._f = f
    self._init_layers(self._n,self._f)

  def _init_layers(self,n,f):
    # Upper branch
    setattr(self,'res'+str(n)+'_1',Residual(f,f))
    # Lower branch
    setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
    setattr(self,'res'+str(n)+'_2',Residual(f,f))
    if n > 1:
      self._init_layers(n-1,f)
    else:
      self.res_center = Residual(f,f)
    setattr(self,'res'+str(n)+'_3',Residual(f,f))
    setattr(self,'unsample'+str(n),Upsample(scale_factor=2))


  def _forward(self,x,n,f):
    # Upper branch
    up1 = x
    up1 = eval('self.res'+str(n)+'_1')(up1)
    # Lower branch
    low1 = eval('self.pool'+str(n)+'_1')(x)
    low1 = eval('self.res'+str(n)+'_2')(low1)
    if n > 1:
      low2 = self._forward(low1,n-1,f)
    else:
      low2 = self.res_center(low1)
    low3 = low2
    low3 = eval('self.'+'res'+str(n)+'_3')(low3)
    up2 = eval('self.'+'unsample'+str(n)).forward(low3)

    return up1+up2

  def forward(self,x):
    return self._forward(x,self._n,self._f)



class HGNet(nn.Module):
  def __init__(self,tune=False):
    super(HGNet,self).__init__()
    self.__conv1 = nn.Conv2d(3,64,1)
    self.__relu1 = nn.ReLU(inplace=True)
    self.__conv2 = nn.Conv2d(64,128,1)
    self.__relu2 = nn.ReLU(inplace=True)
    self.__hg = HourGlass()
    self.__lin = Lin()
    self.__tune = Tune()
    self.tune = tune
  def forward(self,x):
    x = self.__relu1(self.__conv1(x))
    x = self.__relu2(self.__conv2(x))
    x = self.__hg(x)
    x = self.__lin(x)
    if self.tune == True:
      x = x.reshape(x.shape[0],-1)
    x = self.__tune(x)
    return x







