#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from PIL import Image
import os

import densenet

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device_ids = [0, 1, 2, 3]
EPOCH = 300
BATCH_SIZE = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


model = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)
model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)


# state_dict = torch.load('./checkpoint_densenet/model_epoch108_acc0.8839835728952772.pth.tar')
state_dict = None

if state_dict:
    model.load_state_dict(state_dict['state_dict'])


# In[8]:


class MyData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = np.load(data_dir)
        self.data_x = [item[0] for item in self.data]
        self.data_y = [item[1] for item in self.data]
        self.transform = transform
    
    def __getitem__(self, index):
        if self.transform is not None:
            x = Image.fromarray(np.uint8(self.data_x[index]).transpose(1,2,0))
            x = self.transform(x)
        else:
            x = self.data_x[index]
        return x, torch.LongTensor([self.data_y[index]])

    def __len__(self):
        return len(self.data_x)


# In[4]:


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, 'model_epoch{}_acc{}.pth.tar'.format(state['epoch'], state['best_prec1'])))
    print("save the best model to {}".format(os.path.join(filepath, 'model_epoch{}_acc{}.pth.tar'.format(state['epoch'], state['best_prec1']))))


# In[9]:


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
    ])
val_transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])])


# In[10]:


train_dir = '../train_data_new.npy'
validation_dir = '../validation_data_new.npy'
train_loader = DataLoader(dataset=MyData(train_dir, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=MyData(validation_dir, transform=val_transform), batch_size=BATCH_SIZE)


# In[11]:


loss_func = torch.nn.CrossEntropyLoss().cuda()
adam = torch.optim.Adam(model.parameters(),lr=0.001)
rms = torch.optim.RMSprop(model.parameters(), lr=0.001)
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
opt = sgd


# In[12]:


if state_dict:
    start_point = state_dict['epoch']
    max_acc = state_dict['best_prec1']
    # opt.load_state_dict(state_dict['optimizer'])
else:
    start_point = 0
    max_acc = 0


model_save_path = './checkpoint_densenet_new'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

for epoch in range(EPOCH):
    if (epoch - start_point) % 80 == 0 and (epoch - start_point) != 0:
        print('adjust the learning rate')
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.1

    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        opt.zero_grad()
        
        output = model(batch_x)
        loss = loss_func(output, batch_y.squeeze())
        
        pred = torch.max(output.data, 1)[1]
        train_correct = (pred == batch_y.squeeze(1)).sum()
        print('epoch:{}/{}  batch:{}/{}  loss:{:.6f}  acc:{:.4f} max_acc{:.4f}'.format(epoch, EPOCH, batch_idx, 45000//BATCH_SIZE, loss, float(train_correct)/BATCH_SIZE, max_acc))
        
        loss.backward()
        opt.step()
    
    model.eval()
    correct = 0
    for batch_idx, (val_x, val_y) in enumerate(validation_loader):
        val_x, val_y = val_x.cuda(), val_y.cuda()
        
        output = model(val_x)
        pred = torch.max(output.data, 1)[1]
        correct += (pred == val_y.squeeze(1)).sum()
    acc = float(correct) / 5000
    print('epoch:{}/{} acc:{:.4f} max_acc:{:.4f}'.format(epoch, EPOCH, acc, max_acc))
    
    if epoch == 0:
        max_acc = acc
    else:
        if acc > max_acc:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': acc,
                'optimizer': opt.state_dict(),
            }, filepath=model_save_path)
            print('validation accuracy improve from {} to {}.'.format(max_acc, acc))
            max_acc = acc


