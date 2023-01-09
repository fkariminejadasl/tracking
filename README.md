Fish tracking
=============

Installation
------------

Make a wheel:
```bash
pip install build
python -m build
```

Install the package:
```bash
pip install --find-links ~/dev/tracking/dist ftracking -r ~/dev/tracking/requirements.txt
```

Install only requirements:
```bash
pip install -r requirements.txt
```

The minimum pyton version is 3.10. If there is not new version of python, conda is the cleanest way. 
These are only command requires in conda, which are different than python venv. 
```bash
conda create -n ftrack python=3.10 # create virtualenv
conda activate ftrack # activate
conda deactivate # deactivate
conda remove --name ftrack --all # remove the virtualenv
```

## Run a script

`-r` result folder, `-d` for your detection folder, where detections are saved, `-v` is the full path to the video file. The other options can be skipped. For more information, use `--help`.
```bash
python ~/dev/tracking/scripts/track_fishes.py -r /home/fatemeh/results/dataset5 -d /home/fatemeh/data/dataset5/cam1_labels -v /home/fatemeh/data/dataset5/04_07_22_F_2_rect_valid.mp4 --video_bbox 270,100,1800,1200 --fps 1 --total_no_frames 10
```

NB. '-r', '-d' or '-v' could be relative path, if you are current location is. For example, both video file and detections are in the same folder of `cd /home/fatemeh/data/dataset5`. Only need to give the relative path.
```bash
cd /home/fatemeh/data/dataset5
python ~/dev/tracking/scripts/track_fishes.py -r /home/fatemeh/results/dataset5 -d cam1_labels -v 04_07_22_F_2_rect_valid.mp4 --video_bbox 270,100,1800,1200 --fps 1 --total_no_frames 10 --save_name test2
```