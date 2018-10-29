# Homework / MP Solutions for CS598 / IE 534

## Workflow for Bluewaters
* Login
    ```
    ssh <bw-username>@bwbay.ncsa.illinois.edu
    ```
* Mount the remote file system into your local machine
    > Note: on Bluewaters, this approach works much better than `scp` since you don't have to manually `scp` everytime you update a file.
    
    At your local machine
    ```
    mkdir <local-path>
    sshfs -o ssh_command="ssh <bw-username>@bwbay.ncsa.illinois.edu ssh" h2ologin:<remote-path> <local-path>
    ```
* Load interactive mode
    ```
    qsub -I -l gres=ccm -l nodes=1:ppn=16:xk -l walltime=02:00:00
    ```
* Login to the compute node

    In compute node, you can directly run your program.
    ```
    module add ccm
    ccmlogin
    module load bwpy/2.0.0-pre1
    ```
* Submit a job to BW

    This can be done simply via login node.
    ```
    qsub run.pbs
    ```
* Check job status
    ```
    qstat | grep <username>
    ```
* Delete a job
    ```
    qdel <job number>.bw
    ```
* Check your remaining training quota
    ```
    usage
    ```

## Sample `run.pbs` file
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
    