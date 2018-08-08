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
from ct_classifier import RegistrationClassificationDataset
from ct_classifier import Net

torch.set_num_threads(35)
test_dataset = RegistrationClassificationDataset(data_dir="/home/asinha8/sinusct/dicom", label_dir = "/home/asinha8/local/data", dataset_id=8)
testdataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=35)

print('Finished setup')
print('Getting Net()')
net = Net()
net = nn.DataParallel(net, device_ids=[0,1,2,3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print('Loading model')
net.load_state_dict(torch.load('ct_classifier.pt'))

print('Testing')
# test goes here
predictions = csv.writer(open("predictions.csv", 'w'))
testlabels  = csv.writer(open("labels.csv", 'w'))

correct = 0
total = 0
with torch.no_grad():
    for testdata in enumerate(testdataloader, 0):
#	pdb.set_trace()
	output = net(testdata[1]['target'].type(torch.FloatTensor).cuda(), testdata[1]['image'].type(torch.FloatTensor).cuda())
#	print output
	_, predicted = torch.max(output.cpu(), 1)
#	predicted = output.cpu()
#	predicted[predicted > 0] =  1
#	predicted[predicted < 0] = -1
	print predicted#.transpose(0,1)[0,:]
	print testdata[1]['label']
#	predictions.writerow(predicted[:,0].numpy())
	predictions.writerow(predicted.numpy())
	testlabels.writerow(testdata[1]['label'].numpy())
	total += testdata[1]['label'].size(0)
#	correct += (predicted.transpose(0,1)[0,:] == testdata[1]['label'].type(torch.FloatTensor)).sum().item()
	correct += (predicted.type(torch.FloatTensor) == testdata[1]['label'].type(torch.FloatTensor)).sum().item()
print('Accuracy of the network: %f %%' % (100 * correct/total))

