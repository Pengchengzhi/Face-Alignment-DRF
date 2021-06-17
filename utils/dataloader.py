import cv2
from torch.utils.data import Dataset, DataLoader
import json
import torch
import os

class Helen_set(Dataset):
  def __init__(self, dataset_type="train", var=1):
    #######################
    # dataset_type="train",     return full train set (2000 pictures)
    # dataset_type="test",     return full test set (330 pictures)
    # dataset_type="sample",    return small train set (200 pictures)
    # dataset_type="sample_test",  return small train set (30 pictures)

    self.dataset_type = dataset_type
    self.var = var

    self.large_gaussian = torch.zeros((521,521))
    for i in range(521):
      for j in range(521):
        numb = torch.tensor(((i-260)**2+(j-260)**2)/2/(self.var**2))
        self.large_gaussian[i,j] = torch.exp(-numb)
    
    if self.dataset_type == 'train' or self.dataset_type == 'sample':
      pts_path = "datasets/cropped_helen/helen_cropped_train_pts.json"
      img_path = "datasets/cropped_helen/trainset"
    
    else:
      pts_path = "datasets/cropped_helen/helen_cropped_test_pts.json"
      img_path = "datasets/cropped_helen/testset"
    

    with open(pts_path,'r') as helen_pts:
      h_pts = json.load(helen_pts)
    self.keys = list(h_pts.keys())

    if self.dataset_type == 'train' or self.dataset_type == 'test':
      N = len(self.keys)
    elif self.dataset_type == 'sample_test': N = 30
    else: N = 200

    self.imgs = torch.zeros(N,3,256,256)
    self.labels = torch.zeros(N,68,2)

    for i in range(N):
      key = self.keys[i]
      path = os.path.join(img_path, key + '.jpg')
      pic = cv2.imread(path)
      pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
      self.imgs[i,:,:,:] = torch.from_numpy(pic).permute(2,0,1)
      self.labels[i,:,:] = torch.Tensor(h_pts[key])

      if self.dataset_type == 'train':
        if (i+1)%400 == 0 or i == 0:
          print('Loading [%d/%d] pictures......' %(i+1,N))
      elif self.dataset_type == 'test':
        if (i+1)%200 == 0 or i == 0:
          print('Loading [%d/%d] pictures......' %(i+1,N))
      else:
        if (i+1)%100 == 0 or i == 0:
          print('Loading [%d/%d] pictures......' %(i+1,N))


  def get_heatmap(self, idx):
    N = len(self.keys)
    npts = self.labels.shape[1]
    gaussian_map = torch.zeros((npts,256,256))
  
    for p in range(npts):
      x = int(self.labels[idx,p,0])
      y = int(self.labels[idx,p,1])
      gaussian_map[p,:,:] = self.large_gaussian[260-x:516-x,260-y:516-y]
    return gaussian_map

  def __len__(self):
    if self.dataset_type == 'train' or self.dataset_type == 'test':
      return len(self.keys)
    elif self.dataset_type == 'sample_test': return 30
    else: return 200

  def __getitem__(self, idx):
    self.heatmap = self.get_heatmap(idx=idx)
    return (self.imgs[idx,:,:,:], self.labels[idx,:,:], self.heatmap)