import pandas as pd
import numpy as np
import cv2
from PIL import Image
import random

data_dir = 'train.csv'

data_frame = pd.read_csv(data_dir)
print(data_frame.shape)
all_data = []
for i in range(data_frame.shape[0]):
    feature = data_frame.iloc[i, 1:-1]
    all_data.append([np.reshape(np.array(feature, dtype=np.float32), (3,32,32)), data_frame.iloc[i, -1]])

random.shuffle(all_data)
random.shuffle(all_data)

train_data = all_data[:45000]
val_data = all_data[45000:]

train_data = np.array(train_data)
val_data = np.array(val_data)
np.save('train_data_new_new.npy', train_data)
np.save('validation_data_new_new.npy', val_data)
