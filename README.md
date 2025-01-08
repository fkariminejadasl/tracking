Fish tracking
=============

Installation
------------
The minimum pyton version is 3.8. If there is not a new version of python, conda is the cleanest way.
These are only command requires in conda, which are different than python venv.
```bash
conda create -n ftrack python=3.8 -y # create virtualenv
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
pip install -e .
```

Installation via whell
------------
> **IMPORTANT**: There are some issues. This part is not working properly.

<details>
<summary>[Click to expand]</summary>

Make a wheel:
```bash
pip install build
python -m build
```

Install the package:
```bash
pip install --find-links ~/dev/tracking/dist ftracking -r ~/dev/tracking/requirements.txt
```
</details>

## Run a script

The parameters are set in `configs/*.yaml`. The help is given in this file.

There are examples in [notebooks](notebooks) how to setup environment and run the scripts.

### stereo tracking script

The parameters are set in `track.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/track_fishes.py ~/dev/tracking/configs/track.yaml
```

### tracking script

The parameters are set in `stereo_track.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/stereo_track_fishes.py ~/dev/tracking/configs/stereo_track.yaml
```

### match ground-truth stereo tracks

The parameters are set in `match_gt_stereo_tracks.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/match_gt_stereo_tracks.py ~/dev/tracking/configs/match_gt_stereo_tracks.yaml
```

### plot tracks in video

The parameters are set in `plot_tracks.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/plot_tracks_in_video.py ~/dev/tracking/configs/plot_tracks.yaml
```

## Use in a code
> Important: This part is not up to date. But in general works like this code snippet. 

To call the package, simply use:
```python
from tracking import data_association as da, visualize
import pathlib as Path
import cv2

# calculate IOU
print(da.get_iou((0, 0, 4, 2), (2, 1, 3, 2)))

# save video as images
save_path = Path("/home/fatemeh/Downloads/fish/mot_data/images/129_1/im")
video_file = Path("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
visualize.save_images_of_video(save_path, video_file, start_frame=0, end_frame=512, step=256)

# save detections as images
save_path = Path("/home/fatemeh/Downloads/fish/mot_data/images/129_1/det")
video_file = Path("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
dets_file = Path("/home/fatemeh/Downloads/fish/mot_data/yolov8/129_1_dets/gt/gt.txt") # in mot format
visualize.save_images_with_detections_mot(save_path, video_file, dets_file, start_frame=0, end_frame=512, step=256)
# The mot format is Table 4 in [mot20](https://arxiv.org/abs/2003.09003)

# save tracks as images
save_path = Path("/home/fatemeh/Downloads/fish/mot_data/images/129_1/tracks")
video_file = Path("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
track_file = Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt") # in mot format
visualize.save_images_with_tracks(save_path, video_file, track_file, start_frame=0, end_frame=512, step=256)

# load tracks
tracks = da.load_tracks_from_mot_format(track_file)

# compute tracks
det_path = main_path / f"yolo/{vid_name}/obj_train_data"
filename_fixpart = "frame"
start_frame, end_frame, step, format = 0, 256, 8, "06d"
image = cv2.imread(str(main_path/f"images/{vid_name}_frame_{start_frame:{format}}.jpg"))
width, height = image.shape[1], image.shape[0]

tracks = da.compute_tracks(
    det_path, filename_fixpart, width, height, start_frame, end_frame, step, format
)
tracks = da._reindex_tracks(da._remove_short_tracks(tracks))
tracks = da.make_array_from_tracks(tracks)
visualize.save_images_with_tracks(
    main_path/"tracks_hung", vid_file, tracks, start_frame=0, end_frame=256, step=8, format="06d"
)
```
