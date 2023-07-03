Fish tracking
=============

Installation
------------
The minimum pyton version is 3.10. If there is not a new version of python, conda is the cleanest way.
These are only command requires in conda, which are different than python venv.
```bash
conda create -n ftrack python=3.10 # create virtualenv
conda activate ftrack # activate
conda deactivate # deactivate
conda remove --name ftrack --all # remove the virtualenv
```

Install only requirements:
```bash
git clone https://github.com/fkariminejadasl/tracking.git
cd tracking
# here the conda should be activated
pip install -r requirements.txt
```

Installation via whell
------------
> **IMPORTANT**: There are some issues. This part is not working properly.

Make a wheel:
```bash
pip install build
python -m build
```

Install the package:
```bash
pip install --find-links ~/dev/tracking/dist ftracking -r ~/dev/tracking/requirements.txt
```

## Run a script

### tracking script

`-r` result folder, `-d` for your detection folder, where detections are saved, `-v` is the full path to the video file. The other options can be skipped. For more information, use `--help`.
```bash
python ~/dev/tracking/scripts/track_fishes.py -r /home/fatemeh/results/dataset5 -d /home/fatemeh/data/dataset5/cam1_labels -v /home/fatemeh/data/dataset5/04_07_22_F_2_rect_valid.mp4 --video_bbox 270,100,1800,1200 --fps 1 --total_no_frames 10
```

NB. '-r', '-d' or '-v' could be relative path, if you are current location is. For example, both video file and detections are in the same folder of `cd /home/fatemeh/data/dataset5`. Only need to give the relative path.
```bash
cd /home/fatemeh/data/dataset5
python ~/dev/tracking/scripts/track_fishes.py -r /home/fatemeh/results/dataset5 -d cam1_labels -v 04_07_22_F_2_rect_valid.mp4 --video_bbox 270,100,1800,1200 --fps 1 --total_no_frames 10 --save_name test2
```

### match ground-truth stereo tracks

```bash
python ~/dev/tracking/scripts/match_gt_stereo_tracks.py -r ~/Downloads -d ~/Downloads/ -t1 04_07_22_F_2_rect_valid_gt.txt -t2 04_07_22_G_2_rect_valid_gt.txt --save_name my_matches.txt
```

### plot tracks in video

```bash
python ~/dev/tracking/scripts/plot_tracks_in_video.py -v ~/Downloads/vids/129_cam_1.MP4 -t ~/Downloads/vids/mot/129_cam_1.zip -s ~/Downloads/result.mp4 --total_no_frames 2000
```

Association learning
------------

The training is in `scripts\training_al.py`, where dataset, model and training and validation step is in `tracking\association_learning.py`. Some statistics and inference is in `exps\inference_al.py`. Association learning is naively integrated in tracking in `tracking\data_association.py::association_learning_matching`. But this implementation has few bugs mentioned in the text. The main bug is caused by asociating one detection to multiple tracks. The other bug is that time and image is not correct when the last detection of the track is not from the previous frame. 

Data is generated in `exps\data_al_v1.py` and statistics are explained in `data_and_exps.md`. Per (query) detection, 4 frames before and 4 after with positional jitter are cropped to 256x512.  

The association learning still requirs some improvement. Here is the list:
- change the loss from cross entropy to binary cross entropy to make the problem simpler.
- use sin encoding for time (similar as positional encoding from attention is all you need paper.)
- increase the capcity of concatanation encoding from 150 to 2024 or higher. 
- add extra class for unmatched items. This requires to add data without matches. 
- [Most likely not]: aggregate multiple frames (4 previous and 4 after), aggregate their flows. Better temporal encoding should be sufficient, since this step is expensive. 
