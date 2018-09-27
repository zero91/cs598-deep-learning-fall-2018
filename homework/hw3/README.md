# Fall 2018 IE534/CS598:  HW3

**Name**: Ziyu Zhou, 
**NetID**: ziyuz2

------

> HW3: 
>
> Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. Due September 28 at 5:00 PM.

## Test accuracy

**`87.13%`**. 

See the following training and testing output (this is NOT the complete outputs since the outputs are too long):

```
*** Performing data augmentation...
Files already downloaded and verified
Files already downloaded and verified
*** Initializing model...
*** Start training on device cuda...
Training [epoch: 1, batch: 1] loss: 2.505, accuracy: 0.10938
Saving model to disk...
Training [epoch: 1, batch: 2] loss: 2.432, accuracy: 0.10156
Training [epoch: 1, batch: 3] loss: 2.392, accuracy: 0.10938
Training [epoch: 1, batch: 4] loss: 2.355, accuracy: 0.12891
...
Training [epoch: 15, batch: 386] loss: 0.275, accuracy: 0.90953
Training [epoch: 15, batch: 387] loss: 0.275, accuracy: 0.90958
Training [epoch: 15, batch: 388] loss: 0.275, accuracy: 0.90951
Training [epoch: 15, batch: 389] loss: 0.275, accuracy: 0.90960
Training [epoch: 15, batch: 390] loss: 0.275, accuracy: 0.90964
Training [epoch: 15, batch: 391] loss: 0.275, accuracy: 0.90966
Training [finished]
*** Start testing...
Testing [batch: 1] loss: 0.317, accuracy: 0.92000
Testing [batch: 2] loss: 0.332, accuracy: 0.90000
Testing [batch: 3] loss: 0.346, accuracy: 0.88667
...
Testing [batch: 96] loss: 0.397, accuracy: 0.87115
Testing [batch: 97] loss: 0.397, accuracy: 0.87155
Testing [batch: 98] loss: 0.397, accuracy: 0.87153
Testing [batch: 99] loss: 0.398, accuracy: 0.87131
Testing [batch: 100] loss: 0.398, accuracy: 0.87130
Testing [finished] finial accuracy: 0.87130
*** Congratulations! You've got an amazing model now :)
```



## Hyperparameters

The hyperparameters are configured as:

| Hyperparameters  | Value |
| :--------------: | :---: |
|  Learning rate   | 0.001 |
| Number of Epochs |  15   |
|    Batch Size    |  128  |

Please refer to the model architecture section for the filter size and number of channels for each layer.



##Model Architecture

The model is a deep CNN based on VGG16 with some modifications. It contains 18 layers, where each layer is either a *CONV* block or a *POOL*block, except that the last layer is a _FC_ block.

* The _CONV_ block contains a convolutional layer, a batch normalization layer and a ReLU layer
* The _POOL_ block contains a max pooling layer
* The _FC_ block contains a dropout layer and a fully connected layer



| Layer No. | Layer Type | Filter Size \| Padding \| Stride | Input \| Output Channels |
| :-------: | :--------: | :------------------------------: | :----------------------: |
|     0     |   INPUT    |                -                 |            -             |
|     1     |    CONV    |         3 x 3 \| 1 \| 1          |         3 \| 64          |
|     2     |    CONV    |         3 x 3 \| 1 \| 1          |         64 \| 64         |
|     3     |    POOL    |         2 x 2 \| 0 \| 2          |            -             |
|     4     |    CONV    |         3 x 3 \| 1 \| 1          |        64 \| 128         |
|     5     |    CONV    |         3 x 3 \| 1 \| 1          |        128 \| 128        |
|     6     |    POOL    |         2 x 2 \| 0 \| 2          |            -             |
|     7     |    CONV    |         3 x 3 \| 1 \| 1          |        128 \| 256        |
|     8     |    CONV    |         3 x 3 \| 1 \| 1          |        256 \| 256        |
|     9     |    CONV    |         3 x 3 \| 1 \| 1          |        256 \| 256        |
|    10     |    POOL    |         2 x 2 \| 0 \| 2          |            -             |
|    11     |    CONV    |         3 x 3 \| 1 \| 1          |        256 \| 512        |
|    12     |    CONV    |         3 x 3 \| 1 \| 1          |        512 \| 512        |
|    13     |    CONV    |         3 x 3 \| 1 \| 1          |        512 \| 512        |
|    14     |    POOL    |         2 x 2 \| 0 \| 2          |            -             |
|    15     |    CONV    |         3 x 3 \| 1 \| 1          |        512 \| 512        |
|    16     |    CONV    |         3 x 3 \| 1 \| 1          |        512 \| 512        |
|    17     |    CONV    |         3 x 3 \| 1 \| 1          |        512 \| 512        |
|    18     |     FC     |                -                 |            -             |



## Usage

Type `python3.6 main.py <trial_num>` in terminal. If the `trial_num` is not provided, it will be set to `5` by default.

The trials are listed as follows:

```python
trials = [
    [0.01, 50],
    [0.001, 50],
    [0.01, 100],
    [0.001, 100],
    [0.001, 70],
    [0.001, 15]]
```



## Implementation

See the file structure:

```
├── data_tools.py			# Tools to load the CIFAR10 dataset and perform data augmentation
├── main.py					# Major file to execute, containing high level pipelines
├── model.py				# Model architecture implementation
├── test.py					# Model testing
├── train.py				# Model training
└── utils.py				# Tools to save and load checkpoints
```

When you run the program, you will get a `data` folder storing the CIFAR10 dataset. After training, you can find your model checkpoint at `checkpoints/model_state.pt`.



