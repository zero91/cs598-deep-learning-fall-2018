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



### Hyperparameters



### Model Architecture





## Part II: Fine-tune a pre-trained ResNet-18 model



### Test accuracy

```
*** Performing data augmentation...
Files already downloaded and verified
Files already downloaded and verified
*** Initializing pre-trained model...
*** Start training on device cuda...
* Hyperparameters: LR = 0.001, EPOCHS = 40, LR_SCHEDULE = True
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