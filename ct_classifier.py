import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
import scipy.misc
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pdb

class RegistrationClassificationDataset:
    
    def __init__(self, data_dir, label_dir, dataset_id):
	# read in labels
	self.label_dir = label_dir
	self.label_data = loadmat(os.path.join(self.label_dir, 'labels_dataset_%d.mat'%dataset_id))
	self.label_data['groundtruth']['label'][0,:][self.label_data['groundtruth']['label'][0,:] == 0] = np.array( [ 0] ) #-1] )
	self.label_data['groundtruth']['label'][0,:][self.label_data['groundtruth']['label'][0,:] == 1] = np.array( [ 1] )
	
	# read in target image
	self.data_dir = os.path.join(data_dir, 'plm_dataset_%d'%dataset_id)
	self.tgt = nib.load('/home/asinha8/sinusct/dicom/average/Average_ANTsOut.nii.gz')
	self.tgt = torch.unsqueeze(torch.from_numpy(self.tgt.get_fdata()), 0)

    def __len__(self):
	return len(self.label_data['groundtruth']['label'][0,:])

    def __getitem__(self, idx):
	# read in images
	img = nib.load(os.path.join(self.data_dir, self.label_data['groundtruth']['name'][0,idx][0]))
	img = torch.unsqueeze(torch.from_numpy(img.get_fdata()), 0)
	sample = {'target': self.tgt, 'image': img, 'label': torch.from_numpy(np.array(self.label_data['groundtruth']['label'][0,idx]))}
	return sample

class Net(nn.Module):
    def __init__(self):
	super(Net, self).__init__()
	self.conv1 = nn.Conv3d(2, 96, 7, stride=3)
	self.pool  = nn.MaxPool3d(4, 4)
	self.conv2 = nn.Conv3d(96, 192, 5)
	self.conv3 = nn.Conv3d(192, 256, 3)
	self.fc1 = nn.Linear(512,64)#(256, 256)
	self.fc2 = nn.Linear(64,2)#1)#(256, 1)

    def forward(self, y, z):
	x = torch.cat((y,z), 1)
	x = self.pool(F.relu(self.conv1(x)))
	x = self.pool(F.relu(self.conv2(x)))
	x = self.pool(F.relu(self.conv3(x)))
	x = x.view(x.size(0),-1)
	x = self.fc1(x)
	x = F.relu(x)
	x = self.fc2(x)
	return x

#def show_slices(slices):
#    fig,axes = plt.subplots(1,len(slices))
#    for i, slice in enumerate(slices):
#	axes[i].imshow(slice.T, cmap="gray", origin="lower")
#    plt.show()

#show_slices([img.dataobj[256,:,:],
#		img.dataobj[:,256,:],
#		img.dataobj[:,:,271]])
