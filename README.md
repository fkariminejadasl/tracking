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

## Use in a code

To call the package, simply use:
```python
from tracking import data_association as da, visualize
import pathlib as Path
import cv2

# calculate IOU
print(da.get_iou((0, 0, 4, 2), (2, 1, 3, 2)))

# save video as images
vid_name = "16_cam12"
main_path = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids")
vid_file = main_path/f"vids/{vid_name}.mp4"
visualize.save_images_of_video(main_path/"images",  vid_file, step=8)

# save tracks as images
tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
save_path = main_path/"images_tracks"
visualize.save_images_with_tracks(
    save_path, vid_file, tracks, start_frame=0, end_frame=256, step=8, format="06d"
)

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

## Run a script

### tracking script

The parameters are set in `track.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/track_fishes.py ~/dev/tracking/configs/track.yaml
```

### match ground-truth stereo tracks

```bash
python ~/dev/tracking/scripts/match_gt_stereo_tracks.py -r ~/Downloads -d ~/Downloads/ -t1 04_07_22_F_2_rect_valid_gt.txt -t2 04_07_22_G_2_rect_valid_gt.txt --save_name my_matches.txt
```

### plot tracks in video

The parameters are set in `plot_tracks.yaml`. The help is given in this file.
```bash
python ~/dev/tracking/scripts/plot_tracks_in_video.py ~/dev/tracking/configs/plot_tracks.yaml
```
