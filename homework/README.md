# Homework / MP Solutions for CS598 / IE 534

## Homework list

* HW1: [Train a perceptron network from scratch for MNIST dataset](hw1/README.md)
* HW2: [Train a multi-channel CNN from scratch for MNIST dataset](hw2/README.md)
* HW3: [Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset](hw3/README.md)


## Workflow for Bluewaters
1. Login
    ```
    ssh <bw-username>@bwbay.ncsa.illinois.edu
    ```
2. Mount the remote file system into your local machine
    > Note: on Bluewaters, this approach works much better than `scp`.
    At your local machine
    ```
    mkdir <local-path>
    sshfs -o ssh_command="ssh <bw-username>@bwbay.ncsa.illinois.edu ssh" h2ologin:<remote-path> <local-path>
    ```
3. Interactive mode
    ```
    qsub -I -l gres=ccm -l nodes=1:ppn=16:xk -l walltime=02:00:00
    ```
4. Add modules
    ```
    module add ccm
    ccmlogin
    module load bwpy/2.0.0-pre1
    ```
5. Run the job
    ```
    qsub run.pbs
    ```
6. Check job status
    ```
    qstat | grep <username>
    ```
    