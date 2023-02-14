# Access Snellius GPUs

## Setup 

**Create an account**

Create ticket or email servicedesk@surf.nl
https://servicedesk.surf.nl 

https://portal.cua.surf.nl : first copied public key in here (only done once)

**Usage**

Use the Snellius (similar for e.g. sshfs/scp):
```bash
ssh -X username@snellius.surf.nl
```

## Use GPUs

</br>

**Run a job:**

NB. The run file should be executable. Make it executable with `chmod a+x runfile.sh`.
```bash
sbatch runfile.sh
```
e.g. runfile.sh:
```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH -o yolo8_train4_%j.out

echo "cpus per task: $SLURM_CPUS_PER_TASK"
echo "cpus on node $SLURM_CPUS_ON_NODE"
echo "cpus per gpu $SLURM_CPUS_PER_GPU"

echo "start training"
yolo detect train data=/home/username/data/data8_v1/data.yaml model=/home/username/exp/runs/detect/bgr23/weights/best.pt imgsz=1920 batch=8 epochs=100 name=bgr cache=true close_mosaic=0 augment=True rect=False mosaic=1.0 mixup=0.0
echo "end training"
```
More SBATCH options and the "output environmental variables" can be found from the [sbatch help](https://slurm.schedmd.com/sbatch.html). 

</br>

**Check job is running**

User squeue with job id or username. 
```bash
squeue -j jobid
squeue -u username
# squeue with more options
squeue -o "%.10i %.9P %.25j %.8u %.8T %.10M %.9l %.6D %.10Q %.20S %R"
```
If the job is running, it will save the result in the output file with the name specified by `SBATCH -o` option. NB. `%j` in the name replaced by job id. In the example `yolo8_train4_%j.out`, the output file will be olo8_train4_2137977.out. The job id is the id you get after running sbatch.

> **IMPORTANT**</br>
Each person has a limited budget in the unit of SBU (system billing unit). It is basically for a GPU, calculated on this formula: </br>
`sbu = # cores * hours * factor`. This factor is `7.11` for partition `gpu`. If you specify 1 GPU, it is 1/4 node, which has 18 cores. 
For example, the SBU is 1280 for 1 GPU and 10 hours: `18 x 10 x 7.11 = ceil(1279.8) = 1280`. 
So in a `runfile.sh`, the basic slurm settings are as:
```bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
```
> NB. `--cpus-per-gpu` or `--cpus-per-task` is automatically set for 1/4 of node, which in `gpu` partition is 18. For more info, check [Snellius accounting](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+usage+and+accounting), [SBU calculating](https://servicedesk.surf.nl/wiki/display/WIKI/Estimating+SBUs).

</br>

There are more options for variables such as below. You can get the full list from the [sbatch help](https://slurm.schedmd.com/sbatch.html). 

```bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
```

> **NB**: Jobs that require more resources and or running for a long time (walltime: `SBATCH --time`) are not easily scheduled. Try to first test if everything is OK by running one or two epochs, then request for resources. Moreover, estimate the time your experiment runs by roughly calculating how long each epoch takes and multiply by epochs and then increase this time for a bit counting for caching, data transfer.

</br>

**Check finished jobs**

```bash
sacct -j jobid -o "JobID,JobName,MaxRSS,Elapsed"
```
More options are in the [sacct help page](https://slurm.schedmd.com/sacct.html).

</br>

**Run tensorboard from remote computer**
Connect to Snellius and map your local port to remote port:
```bash
ssh -X username@snellius.surf.nl -L your_local_port:127.0.0.1:remote_port
```
In Snellius machine, run tensorboard with the remote port:
```bash
tensorboard --logdir_spec=18:/home/username/exp18,12:/home/username/exp12 --port remote_port # remote_port = 60011
```
Now, in the local machine, run `http://localhost:local_port`, e.g. `http://localhost:36999`. 

</br>

**Useful commands**

- `sbatch`: run a job
- `squeue`: show the status of the job
- `scancel`: cancel the job. 
- `scontrol`: show detailed job information
- `sacct`: get statistics on completed jobs
- `accinfo` `accuse`, `budget-overview`: show how much credite is left (Snellius commands)

Some examples are given in [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands). 

</br>

**Useful links:**
- [Snellius accounting](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+usage+and+accounting)
- [SBU calculating](https://servicedesk.surf.nl/wiki/display/WIKI/Estimating+SBUs)
- [Example job scripts](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts)
- [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands)
- [Squeue help](https://slurm.schedmd.com/squeue.html): just use `squeue --help`
