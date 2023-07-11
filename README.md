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

print(da.get_iou((0, 0, 4, 2), (2, 1, 3, 2)))

vid_name = "16_cam12"
main_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids")
vid_path = main_dir/f"vids/{vid_name}.mp4"
visualize.save_video_as_images(main_dir/"images",  vid_path, step=8)

tracks = da.load_tracks_from_mot_format(main_dir / f"mots/{vid_name}.zip")
save_path = main_dir/"images_tracks"
visualize.save_video_with_tracks_as_images(
    save_path, vid_path, tracks, start_frame=0, end_frame=256, step=8, format="06d"
)
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
