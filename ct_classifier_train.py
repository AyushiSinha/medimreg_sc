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
import datetime

torch.set_num_threads(35)

net = Net()
net = nn.DataParallel(net, device_ids=[0,1,2,3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = nn.CrossEntropyLoss() #HingeEmbeddingLoss() #MarginRankingLoss() 
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3) #ASGD(net.parameters()) 
print("Finished setup")

writer = csv.writer(open("loss.csv", "w"))
for d_id in range(3,8):
    print('Dataset %d' %d_id)
    ct_dataset = []
    ct_dataset = RegistrationClassificationDataset(data_dir="/home/asinha8/sinusct/dicom", label_dir="/home/asinha8/local/data", dataset_id=d_id)
    dataloader = DataLoader(ct_dataset, batch_size=4, shuffle=True, num_workers = 35)
    for epoch in range(5):
	running_loss = 0.0
	for i, data in enumerate(dataloader,0):
	    target = []
	    inputs = []
	    labels = []
	    target, inputs, labels = data
#	    print(data['label'].size())
	    optimizer.zero_grad()
	    outputs = net(data['target'].type(torch.FloatTensor), data['image'].type(torch.FloatTensor))
#	    print outputs
	    loss = criterion(outputs.cpu(), data['label'].type(torch.LongTensor))
	    loss.backward()
	    optimizer.step()

	    running_loss += loss.item()
	    n = 10
	    if i%n==(n-1):
		print('[%d, %5d] loss: %.3f' %( epoch + 1, i+1, running_loss / (float)(n) ))
		writer.writerow( [running_loss/(float)(n)] )
		running_loss = 0.0
print("Finished training")
torch.save(net.state_dict(), 'ct_classifier.pt')

