# Fall 2018 IE534/CS598:  HW3

**Name**: Ziyu Zhou, 
**NetID**: ziyuz2

------

> HW3: 
>
> Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. Due September 28 at 5:00 PM.

## Test accuracy

**`98.2%`**. See the following training and testing output:

```

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

Type `python3 main.py` in terminal.



## Implementation

See the file structure:

```
├── checkpoints
├── data
├── data_tools.py
├── main.py
├── model.py
├── remote
├── test.py
├── train.py
└── utils.py
```





