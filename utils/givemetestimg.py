import argparse, os, sys
from PIL import Image
import numpy as np
import torch

def testimgbatch(path):
  # Load img1
  with open(os.path.join(path, '1051618982_1' + '.pts'), "r") as f:
    afw_test_points = f.read().strip().split("\n")
    landmarks = []
    for location_str in afw_test_points[3:-1]:
      location = location_str.split(" ")
      landmarks.append((float(location[0]), float(location[1])))
  afw_test_points = landmarks



  afw_test_x = [afw_test_points[i][0]-408 for i in range(68)]
  afw_test_y = [afw_test_points[i][1]-116 for i in range(68)]



  afw_test_img = Image.open(os.path.join(path, '1051618982_1' + '.jpg'))

  # crop
  afw_test_img_array = np.array(afw_test_img)
  afw_test_nimg_array = afw_test_img_array[0:500, 400:800]
  afw_test_nimg = Image.fromarray(afw_test_nimg_array)


  # Load img2
  with open(os.path.join(path, '134212_2' + '.pts'), "r") as f:
    afw_test2_points = f.read().strip().split("\n")
    landmarks2 = []
    for location2_str in afw_test2_points[3:-1]:
      location2 = location2_str.split(" ")
      landmarks2.append((float(location2[0]), float(location2[1])))
  afw_test_points2 = landmarks2




  afw_test_x2 = [afw_test_points2[i][0]-100 for i in range(68)]
  afw_test_y2 = [afw_test_points2[i][1]-100 for i in range(68)]




  afw_test_img2 = Image.open(os.path.join(path, '134212_2' + '.jpg'))

  # crop
  afw_test_img2_array = np.array(afw_test_img2)
  afw_test_nimg2_array = afw_test_img2_array[100:484, 100:484]
  afw_test_nimg2 = Image.fromarray(afw_test_nimg2_array)


  # to tensor
  img_tensor = torch.stack((torch.tensor(afw_test_nimg_array[116:500,8:392]),torch.tensor(afw_test_nimg2_array)),dim=0)
  img_tensor = img_tensor.permute(0,3,1,2) # Change to (C,H,W)
  img_tensor = img_tensor.float()
  return img_tensor,(afw_test_x,afw_test_y,afw_test_x2,afw_test_y2)
