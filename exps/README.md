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
yolo detect train data=/home/username/data/data8_v1/data.yaml model=/home/username/exp/runs/detect/bgr12/weights/best.pt rect=true imgsz=1920 batch=16 epochs=400 name=bgr cache=true
echo "end training"
```
More SBATCH options and the "output environmental varibles" can be found from the [sbatch help](https://slurm.schedmd.com/sbatch.html). 

**Check job is running**

User squeue with job id or username. 
```bash
squeue -j jobid
squeue -u username
# squeue with more options
squeue -u username -o "%.18i %.9P %.18j %.8u %.2t %.10M %.6D %.10R %.20S %.4p"
squeue -o "%.10i %.9P %.25j %.8u %.8T %.10M %.9l %.6D %.10Q %.20S %R"
```
If the job is running, it will save the result in output file with the name specified by `SBATCH -o` option. NB. `%j` in the name replaced by job id. In the example `yolo8_train4_%j.out`, the output file will be olo8_train4_2137977.out. The job id is the id you get after running sbatch. 

> **IMPORTANT**</br>
Each person has a limited budget in the unit of SBU (system billing unit). It is basically for a GPU, calcuated on this formula: </br>
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


## Useful commands

- `sbatch`: run a job
- `squeue`: show the status of the job
- `scancel`: cancel the job. 
- `scontrol`: show detailed job information
- `sacct`: get statistics on completed jobs
- `accinfo` `accuse`, `budget-overview`: show how much credite is left (Snellius commands)

Some examples are given in [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands). 

**Run tensorboard from remote computer**
Connect to Snellius and map your local port to remote port:
```bash
ssh -X username@snellius.surf.nl -L your_local_port:127.0.0.1:remote_port
```
In Snellius machine, run tensorboar with the remote port:
```bash
tensorboard --logdir_spec=18:/home/username/exp18,12:/home/username/exp12 --port remote_port # remote_port = 60011
```
Now, in the local machine, run `http://localhost:local_port`, e.g. `http://localhost:36999`. 

**Useful links:**
- [Snellius accounting](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+usage+and+accounting)
- [SBU calculating](https://servicedesk.surf.nl/wiki/display/WIKI/Estimating+SBUs)
- [Example job scripts](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts)
- [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands)
- [Squeue help](https://slurm.schedmd.com/squeue.html): just use `squeue --help`



# Data

**data8_v1** </br>
[description]: Every 8th frame take samples only for F, G every frame. 231_cam_1 used for val. </br>
[code]: prepare_data_for_yolo_all(save_path, videos_main_path, labels_main_path) in `data_prepration_v1.py` </br>
[labels]: zip files in yolo format
[vids]: 04_07_22_F_2_rect_valid, 04_07_22_G_2_rect_valid, 129_cam_1, 129_cam_2, 161_cam_1, 161_cam_2, 183_cam_1, 183_cam_2, 231_cam_1, 231_cam_2, 261_cam_1, 349_cam_1, 349_cam_2, 406_cam_1, 406_cam_2 </br>

 <!-- (#594 images: 2704 x 1520 orig) -> #4092/400 (1024 x 576 train)/(960 x 540 val) -->

# Data Statistics

- Code for this statistics generated by `exps/data_stats.py`.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vid</th>
      <th>imsize</th>
      <th>#frames</th>
      <th>fps</th>
      <th>#dets</th>
      <th>#tracks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04_07_22_F_2_rect_valid</td>
      <td>1220x2098</td>
      <td>600</td>
      <td>30</td>
      <td>26413</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04_07_22_G_2_rect_valid</td>
      <td>1220x2098</td>
      <td>600</td>
      <td>30</td>
      <td>22633</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>129_cam_1</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>30201</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129_cam_2</td>
      <td>1080x1920</td>
      <td>3118</td>
      <td>240</td>
      <td>30649</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161_cam_1</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>35693</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>161_cam_2</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>38349</td>
      <td>23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>183_cam_1</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>105580</td>
      <td>48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>183_cam_2</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>86160</td>
      <td>47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>231_cam_1</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>68671</td>
      <td>32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>231_cam_2</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>82792</td>
      <td>35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>261_cam_1</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>46718</td>
      <td>47</td>
    </tr>
    <tr>
      <th>11</th>
      <td>349_cam_1</td>
      <td>1080x1920</td>
      <td>3118</td>
      <td>240</td>
      <td>27195</td>
      <td>10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>349_cam_2</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>27453</td>
      <td>9</td>
    </tr>
    <tr>
      <th>13</th>
      <td>406_cam_1</td>
      <td>1080x1920</td>
      <td>3118</td>
      <td>240</td>
      <td>27011</td>
      <td>10</td>
    </tr>
    <tr>
      <th>14</th>
      <td>406_cam_2</td>
      <td>1080x1920</td>
      <td>3117</td>
      <td>240</td>
      <td>26554</td>
      <td>10</td>
    </tr>
  </tbody>
</table>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vid</th>
      <th>track_id:track_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04_07_22_F_2_rect_valid</td>
      <td>0:600,1:600,2:600,3:600,4:600,5:600,6:600,7:600,8:600,9:600,10:600,11:600,12:600,13:600,14:600,15:600,16:600,17:600,18:600,19:600,20:600,21:254,22:600,23:600,24:600,25:600,26:600,27:310,28:600,29:600,30:600,31:600,32:600,33:600,34:600,35:600,36:600,37:600,38:600,39:600,40:600,41:600,42:600,43:600,44:599,45:50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04_07_22_G_2_rect_valid</td>
      <td>0:600,1:600,2:600,3:600,4:600,5:164,6:600,7:600,8:600,9:600,10:600,11:600,12:600,13:600,14:213,15:600,16:600,17:600,18:600,19:600,20:600,21:600,22:600,23:600,24:600,25:586,26:600,27:600,28:600,29:600,30:600,31:570,32:600,33:600,34:600,35:600,36:600,37:536,38:443,39:321,40:270,41:164,42:134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>129_cam_1</td>
      <td>0:3117,1:3117,2:316,3:3117,4:3117,5:3117,6:3117,7:3117,8:3117,9:1075,10:1473,11:1101,12:848,13:251,14:201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129_cam_2</td>
      <td>0:3118,1:3118,2:3118,3:334,4:3118,5:3118,6:3118,7:3118,8:3118,9:1064,10:1509,11:1202,12:844,13:551,14:201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161_cam_1</td>
      <td>0:3117,1:2791,2:1598,3:3117,4:2719,5:2507,6:2563,7:550,8:3117,9:1833,10:3117,11:3117,12:2077,13:322,14:1080,15:668,16:261,17:423,18:319,19:272,20:125</td>
    </tr>
    <tr>
      <th>5</th>
      <td>161_cam_2</td>
      <td>0:1936,1:1650,2:1666,3:3117,4:810,5:3117,6:2496,7:506,8:3117,9:3117,10:3117,11:1709,12:1733,13:800,14:1548,15:2090,16:937,17:1285,18:837,19:961,20:684,21:665,22:451</td>
    </tr>
    <tr>
      <th>6</th>
      <td>183_cam_1</td>
      <td>0:270,1:2100,2:1500,3:3117,4:3117,5:3117,6:3117,7:3117,8:3117,9:3117,10:3117,11:3117,12:3117,13:3117,14:3117,15:3117,16:3117,17:2070,18:3117,19:3117,20:3117,21:3117,22:3117,23:2565,24:2505,25:1974,26:3117,27:3117,28:1607,29:2666,30:3117,31:3117,32:1320,33:1613,34:1989,35:1947,36:1286,37:1228,38:944,39:804,40:709,41:702,42:688,43:686,44:684,45:682,46:681,47:669</td>
    </tr>
    <tr>
      <th>7</th>
      <td>183_cam_2</td>
      <td>0:2217,1:3117,2:3117,3:3117,4:2047,5:3117,6:3117,7:2013,8:3117,9:3117,10:3117,11:2520,12:2567,13:3117,14:3117,15:3117,16:1917,17:3117,18:3117,19:3117,20:3117,21:1533,22:3117,23:3117,24:1053,25:3117,26:266,27:480,28:270,29:1854,30:1794,31:323,32:1104,33:767,34:714,35:696,36:134,37:683,38:677,39:675,40:674,41:671,42:654,43:516,44:467,45:437,46:331</td>
    </tr>
    <tr>
      <th>8</th>
      <td>231_cam_1</td>
      <td>0:3117,1:3117,2:3117,3:3117,4:3117,5:3117,6:3117,7:3117,8:3117,9:1920,10:1139,11:3117,12:3117,13:3117,14:3117,15:2946,16:1019,17:3057,18:3055,19:296,20:215,21:2782,22:2722,23:2542,24:414,25:2422,26:836,27:1458,28:670,29:425,30:176,31:56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>231_cam_2</td>
      <td>0:3117,1:3117,2:3117,3:3117,4:3117,5:3117,6:3117,7:3117,8:1852,9:776,10:3117,11:3117,12:3117,13:3117,14:3117,15:3117,16:3117,17:1453,18:3117,19:3117,20:3117,21:210,22:3117,23:2924,24:3117,25:3027,26:2769,27:2679,28:2581,29:445,30:210,31:522,32:539,33:404,34:61</td>
    </tr>
    <tr>
      <th>10</th>
      <td>261_cam_1</td>
      <td>0:850,1:1612,2:3117,3:3117,4:2543,5:1240,6:120,7:1030,8:1618,9:120,10:3117,11:2658,12:180,13:377,14:650,15:2950,16:3117,17:1658,18:230,19:1128,20:1685,21:929,22:746,23:171,24:685,25:680,26:644,27:671,28:670,29:661,30:378,31:655,32:654,33:163,34:651,35:647,36:641,37:638,38:629,39:624,40:620,41:490,42:90,43:331,44:231,45:211,46:91</td>
    </tr>
    <tr>
      <th>11</th>
      <td>349_cam_1</td>
      <td>0:1371,1:3118,2:3118,3:3118,4:3118,5:3118,6:3118,7:2918,8:2619,9:1579</td>
    </tr>
    <tr>
      <th>12</th>
      <td>349_cam_2</td>
      <td>0:3117,1:3117,2:3117,3:3117,4:3117,5:3117,6:3117,7:3117,8:2517</td>
    </tr>
    <tr>
      <th>13</th>
      <td>406_cam_1</td>
      <td>0:3118,1:3118,2:3118,3:3118,4:3118,5:3118,6:3118,7:524,8:3118,9:1543</td>
    </tr>
    <tr>
      <th>14</th>
      <td>406_cam_2</td>
      <td>0:3117,1:3117,2:3117,3:3117,4:3117,5:701,6:3117,7:3117,8:3117,9:917</td>
    </tr>
  </tbody>
</table>