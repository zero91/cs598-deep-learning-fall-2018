------
# Fall 2018 IE534/CS598:  HW4

**Name**: Ziyu Zhou, 
**NetID**: ziyuz2

------

> * Build the Residual Network speciﬁed in Figure 1 and achieve at least 60% test accuracy.
>
>   In the homework, you should deﬁne your “Basic Block” as shown in Figure 2. For each weight layer, it should contain 3 × 3 ﬁlters for a speciﬁc number of input channels and output channels. The output of a sequence of ResNet basic blocks goes through a max pooling layer with your own choice of ﬁlter size, and then goes to a fully-connected layer. The hyperparameter speciﬁcation for each component is given in Figure 1. Note that the notation follows the notation in He et al. (2015).
>
> * Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.

## Part I: Build the required Residual Network

### Test accuracy
**`60.78%`**.

See the following training and testing output

```
*** Performing data augmentation...
Files already downloaded and verified
Files already downloaded and verified
*** Initializing model...
*** Start training on device cuda...
* Hyperparameters: LR = 0.001, EPOCHS = 40, LR_SCHEDULE = True
Training [epoch: 1] loss: 3.898, accuracy: 0.10184
Testing [finished] accuracy: 0.11780
Training [epoch: 2] loss: 3.258, accuracy: 0.20254
Testing [finished] accuracy: 0.24320
Training [epoch: 3] loss: 2.839, accuracy: 0.28422
Testing [finished] accuracy: 0.30210
Training [epoch: 4] loss: 2.507, accuracy: 0.35088
Testing [finished] accuracy: 0.35110
Training [epoch: 5] loss: 2.252, accuracy: 0.40424
Testing [finished] accuracy: 0.38980
Training [epoch: 6] loss: 2.053, accuracy: 0.44694
Testing [finished] accuracy: 0.43320
Training [epoch: 7] loss: 1.886, accuracy: 0.48786
Testing [finished] accuracy: 0.46730
Training [epoch: 8] loss: 1.745, accuracy: 0.52036
Testing [finished] accuracy: 0.50420
Training [epoch: 9] loss: 1.624, accuracy: 0.54868
Testing [finished] accuracy: 0.52490
Training [epoch: 10] loss: 1.522, accuracy: 0.57464
Testing [finished] accuracy: 0.52490
Training [epoch: 11] loss: 1.435, accuracy: 0.59412
Testing [finished] accuracy: 0.53310
Training [epoch: 12] loss: 1.345, accuracy: 0.61246
Testing [finished] accuracy: 0.53460
Training [epoch: 13] loss: 1.272, accuracy: 0.63438
Testing [finished] accuracy: 0.54780
Training [epoch: 14] loss: 1.207, accuracy: 0.65014
Testing [finished] accuracy: 0.54630
Training [epoch: 15] loss: 1.139, accuracy: 0.66864
Testing [finished] accuracy: 0.58350
Training [epoch: 16] loss: 1.083, accuracy: 0.68026
Testing [finished] accuracy: 0.58020
Training [epoch: 17] loss: 1.020, accuracy: 0.69806
Testing [finished] accuracy: 0.59200
Training [epoch: 18] loss: 0.966, accuracy: 0.71434
Testing [finished] accuracy: 0.59310
Training [epoch: 19] loss: 0.911, accuracy: 0.72678
Testing [finished] accuracy: 0.59690
Training [epoch: 20] loss: 0.865, accuracy: 0.73824
Testing [finished] accuracy: 0.59320
Training [epoch: 21] loss: 0.808, accuracy: 0.75518
Testing [finished] accuracy: 0.58260
Training [epoch: 22] loss: 0.768, accuracy: 0.76566
Testing [finished] accuracy: 0.58350
Training [epoch: 23] loss: 0.727, accuracy: 0.77670
Testing [finished] accuracy: 0.59340
Training [epoch: 24] loss: 0.681, accuracy: 0.78986
Testing [finished] accuracy: 0.60420
Training [epoch: 25] loss: 0.637, accuracy: 0.80136
Testing [finished] accuracy: 0.61030
Training [epoch: 26] loss: 0.600, accuracy: 0.81386
Testing [finished] accuracy: 0.60130
Training [epoch: 27] loss: 0.567, accuracy: 0.82120
Testing [finished] accuracy: 0.61750
Training [epoch: 28] loss: 0.526, accuracy: 0.83510
Testing [finished] accuracy: 0.60180
Training [epoch: 29] loss: 0.516, accuracy: 0.83742
Testing [finished] accuracy: 0.60980
Training [epoch: 30] loss: 0.482, accuracy: 0.84676
Testing [finished] accuracy: 0.58970
Training [epoch: 31] loss: 0.452, accuracy: 0.85512
Testing [finished] accuracy: 0.62190
Training [epoch: 32] loss: 0.428, accuracy: 0.86356
Testing [finished] accuracy: 0.60910
Training [epoch: 33] loss: 0.500, accuracy: 0.83992
Testing [finished] accuracy: 0.60500
Training [epoch: 34] loss: 0.377, accuracy: 0.87944
Testing [finished] accuracy: 0.60250
Training [epoch: 35] loss: 0.351, accuracy: 0.88846
Testing [finished] accuracy: 0.60830
Training [epoch: 36] loss: 0.345, accuracy: 0.88918
Testing [finished] accuracy: 0.59210
Training [epoch: 37] loss: 0.331, accuracy: 0.89140
Testing [finished] accuracy: 0.59750
Training [epoch: 38] loss: 0.306, accuracy: 0.90064
Testing [finished] accuracy: 0.59470
Training [epoch: 39] loss: 0.293, accuracy: 0.90454
Testing [finished] accuracy: 0.60860
Training [epoch: 40] loss: 0.277, accuracy: 0.90980
Testing [finished] accuracy: 0.60780
Training [finished]
*** Start testing...
Testing [finished] accuracy: 0.60780
*** Congratulations! You've got an amazing model now :)
```

### Hyperparameters



### Model Architecture





## Part II: Fine-tune a pre-trained ResNet-18 model



### Test accuracy
**`72.03%`**.

See the following training and testing output

```
*** Performing data augmentation...
Files already downloaded and verified
Files already downloaded and verified
*** Initializing pre-trained model...
*** Start training on device cuda...
* Hyperparameters: LR = 0.001, EPOCHS = 10, LR_SCHEDULE = True
Training [epoch: 1] loss: 1.748, accuracy: 0.51720
Testing [finished] accuracy: 0.58120
Training [epoch: 2] loss: 1.077, accuracy: 0.68164
Testing [finished] accuracy: 0.64220
Training [epoch: 3] loss: 0.828, accuracy: 0.74998
Testing [finished] accuracy: 0.68730
Training [epoch: 4] loss: 0.667, accuracy: 0.79380
Testing [finished] accuracy: 0.69840
Training [epoch: 5] loss: 0.540, accuracy: 0.83136
Testing [finished] accuracy: 0.71700
Training [epoch: 6] loss: 0.441, accuracy: 0.85970
Testing [finished] accuracy: 0.72000
Training [epoch: 7] loss: 0.362, accuracy: 0.88270
Testing [finished] accuracy: 0.71870
Training [epoch: 8] loss: 0.306, accuracy: 0.90268
Testing [finished] accuracy: 0.71210
Training [epoch: 9] loss: 0.255, accuracy: 0.91652
Testing [finished] accuracy: 0.72120
Training [epoch: 10] loss: 0.212, accuracy: 0.93068
Testing [finished] accuracy: 0.72030
Training [finished]
*** Start testing...
Testing [finished] accuracy: 0.72030
*** Congratulations! You've got an amazing model now :)
```



### Hyperparameters



## Usage

### Run the program

To train the customized ResNet model built in *Part I*, run

```
python3 main.py
```

To fine tune the pretrained model in *Part II*, run

```
python3 main.py --fine_tune True
```



### Configurations

Type `python3 main.py --help` to see the available configuration flags, like so

```
$ python3 main.py --help
usage: main.py [-h] [--fine_tune FINE_TUNE] [--lr LR] [--epochs EPOCHS]
               [--lr_schedule LR_SCHEDULE] [--load_checkpoint LOAD_CHECKPOINT]
               [--show_sample_image SHOW_SAMPLE_IMAGE] [--debug DEBUG]
               [--data_path DATA_PATH]

Training ResNet on CIFAR100

optional arguments:
  -h, --help            show this help message andexit
  --fine_tune FINE_TUNE
                        fine-tune pretrained model
  --lr LR               learning rate
  --epochs EPOCHS       number of training epochs
  --lr_schedule LR_SCHEDULE
                        perform lr shceduling
  --load_checkpoint LOAD_CHECKPOINT
                        resume from checkpoint
  --show_sample_image SHOW_SAMPLE_IMAGE
                        display data insights
  --debug DEBUG         using debug mode
  --data_path DATA_PATH
                        path to store data
```



## Implementation

See the file structure:

```
.
├── README.md
├── data_tools.py
├── fine_tune.py
├── layer.py
├── main.py
├── model.py
├── test.py
├── train.py
└── utils.py
```

When you run the program, you will get a `data` folder storing the CIFAR100 dataset. After training, you can find your model checkpoint at `checkpoints/model_state.pt`.