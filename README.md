# cs598-deep-learning-fall-2018
This repo contains assignments of [CS598 / IE534 Deep Learning @ UIUC](https://courses.engr.illinois.edu/ie534/fa2018/) (Fall 2018 semester).



## Homework (Machine Problems)

* HW1: [Train a perceptron network from scratch for MNIST dataset](homework/hw1/README.md)
* HW2: [Train a multi-channel CNN from scratch for MNIST dataset](homework/hw2/README.md)
* HW3: [Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset](homework/hw3/README.md)
* HW4: [Implement a deep residual neural network for CIFAR100](homework/hw4/README.md)
* HW5: [Implement a deep learning model for image ranking](homework/hw5/README.md)
* HW6: [Generative adversarial networks (GANs)](homework/hw6/README.md)
* HW7: [Natural Language Processing A](homework/hw7/README.md)
* HW8: [Natural Language Processing B](homework/hw7)
* HW9: [Video recognition I](homework/hw9)



## Notes

### Workflow for Bluewaters

- Login

  ```
  ssh <bw-username>@bwbay.ncsa.illinois.edu
  ```

- Mount the remote file system into your local machine

  > Note: on Bluewaters, this approach works much better than `scp` since you don't have to manually `scp` everytime you update a file.

  At your local machine

  ```
  mkdir <local-path>
  sshfs -o ssh_command="ssh <bw-username>@bwbay.ncsa.illinois.edu ssh" h2ologin:<remote-path> <local-path>
  ```

- Load interactive mode

  ```
  qsub -I -l gres=ccm -l nodes=1:ppn=16:xk -l walltime=02:00:00
  ```

- Login to the compute node

  In compute node, you can directly run your program.

  ```
  module add ccm
  ccmlogin
  module load bwpy/2.0.0-pre1
  ```

- Submit a job to BW using a `.pbs` file

  This can be done simply via login node. See below for a template.

  ```
  qsub run.pbs
  ```

- Check job status

  ```
  qstat | grep <username>
  ```

- Delete a job

  ```
  qdel <job number>.bw
  ```

- Check your remaining training quota

  ```
  usage
  ```

### Sample `.pbs` file

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N <job name>
#PBS -l walltime=40:00:00
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -M <your email to receive messages about your job>
cd <path to the folder that your program locates>
. /opt/modules/default/init/bash
module load bwpy/2.0.0-pre1
module load cudatoolkit
aprun -n 1 -N 1 python3.6 <program name>
```