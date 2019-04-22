import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F
import densenet

test_dir = '../test.csv'

test_frame = pd.read_csv(test_dir)
print(test_frame.shape)
all_data = []
for i in range(test_frame.shape[0]):
    feature = test_frame.iloc[i, 1:]
    all_data.append(np.reshape(np.array(feature, dtype=np.float32), (3,32,32)))



os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device_ids = [0, 1, 2, 3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)
model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)


state_dict = torch.load('./checkpoint_densenet_new/model_epoch269_acc0.9234.pth.tar')
model.load_state_dict(state_dict['state_dict'])

test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])])

model.eval()
test_result = []
with torch.no_grad():
    for i in range(len(all_data)):
        print(i)
        x = Image.fromarray(np.uint8(np.transpose(all_data[i], (1,2,0))))
        test_x = test_transform(x)
        test_x = test_x.reshape((1,3,32,32))
        output = model(test_x)
        output = F.log_softmax(output)
        print(output.data)
        pred = torch.max(output.data, 0)[1]
        test_result.append(int(pred))

ID = [i for i in range(10000)]
result_dict = {'ID':ID, 'Category':test_result}
columns = ['ID', 'Category']
df = pd.DataFrame(result_dict)
df.to_csv('wen_result (8).csv', index=False, columns=columns)
