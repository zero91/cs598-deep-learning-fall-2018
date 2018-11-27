"""
Once again, we can achieve better performance on the test dataset 
by using the full video instead of just 16 frames. 
Use the code provided in part 1 for testing as a starting point 
to create similar code for testing the 3D ResNet on the full video 
sequences. There can be a couple of approaches.

* Process subsequences of length 16 and average the final output. 
This is the most basic way and easiest to implement but not the best way. 
Notice there is a difference between processing every 16 frames 
compared with processing every subsequence of length 16. 
There are overlapping subsequences. 
If you are unsure of how to do the next option, 
it may be smart to start here just to get something working to see 
what to expect.

* Pass the full sequence in as input. 
When we defined the model, we set sample_duration=16. 
This is the minimum length for training but not the max length. 
With 16 frames, the temporal dimension has been reduced to size 1. 
That is, the output after model.layer4[0] is [batch_size,2048,1,7,7] 
where 2048 is the number of channels, 1 is the temporal dimension, 
and 7x7 is the spatial dimension. 
By specifying sample_duration=16, model.avgpool() is created with 
the knowledge that average pooling will need to be performed over 
the 1x7x7 dimensions. If a sequence of length 32 was passed into 
the network, the output of model.layer4[0] would be [batch_size,2048,2,7,7]. 
If average pooling is performed over the 2x7x7 dimensions, 
then this can be passed through the model.fc layer to get a single 
prediction for the entire sequence of length 32. 
For each video, perform the appropriately sized averaging pooling 
such that the full sequence can be passed through the network. 
You may want to look at the model definition resnet_3d.py for ideas.

Save the output for each video 
(this should be a single prediction as opposed to a 
prediction for each frame). Also save the confusion matrix.
"""

import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadSequence

import h5py
import cv2

from multiprocessing import Pool

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 32
lr = 0.0001
num_of_epochs = 10
num_of_frames = 16

model = torch.load('3d_resnet.model')
model.cuda()

data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

##### save predictions directory
prediction_directory = 'UCF-101-predictions-3d/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)
model.eval()

for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    # data = np.zeros((nFrames,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)
    num_sequence = nFrames // num_of_frames
    data = np.zeros(
        (num_sequence, 3, num_of_frames, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32
    )

    for j in range(num_sequence):
        for k in range(j * num_of_frames, (j + 1) * num_of_frames):
            frame = h['video'][k]
            frame = frame.astype(np.float32)
            frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
            frame = frame/255.0
            frame = (frame - mean)/std
            frame = frame.transpose(2,0,1)
            data[j, :, k - j * num_of_frames, :, :] = frame
    h.close()

    prediction = np.zeros((num_sequence, NUM_CLASSES), dtype=np.float32)

    loop_i = list(range(0, num_sequence, 10))
    loop_i.append(num_sequence)

    for j in range(len(loop_i)-1):
        data_batch = data[loop_i[j]:loop_i[j+1]]

        with torch.no_grad():
            # x = np.asarray(data_batch,dtype=np.float32)
            # x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            # output = model(x)

            x = np.asarray(data_batch, dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            # h = model.layer4[1](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)

        prediction[loop_i[j]:loop_i[j+1]] = output.cpu().numpy()
    
    filename = filename.replace(data_directory+'UCF-101-hdf5/', prediction_directory)
    if(not os.path.isfile(filename)):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions', data=prediction)

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))

    # create a vector of prob of classfying this video into each class
    prediction = np.sum(np.log(prediction), axis=0)   # sum all rows (all sequences)
    argsort_pred = np.argsort(-prediction)[0:10]   # sort from large to small

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))
    
number_of_examples = np.sum(confusion_matrix,axis=1)   # num examples of videos classfied into each class
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    # 1. name of a class, 
    # 2. num of samples of this class being classfied correctly, 
    # 3. total num of samples classfied as this class
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('3d_conv_confusion_matrix.npy',confusion_matrix)