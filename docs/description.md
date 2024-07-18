Multistage tracking
===================

Multistage tracking consists of two stages, where (nearly) occluded detections and the rest are treated separately. In the first stage, unoccluded areas were matched by IOU using the Hungarian algorithm. In the second stage, the (nearly) occluded area matched by their deep-feature cosine similarity and their IOU. There are several improvements made specific to fish data. These are:

- To group detections based on occlusion, in each image, (nearly occluded) regions are identified. Then matching groups were identified based on inter-frame IOU. The near occlusion considers detections within distance of 3 pixels as nearly occluded.
- Improved Hungarian method used for stage1 (remove rows and columns before matching). The cost matrix (affinity matrix) below a certain threshold. If all the rows or all columns are above the threshold, they will be removed.
- Non-maximum suppression (NMS) on fully occluded detections. If detections are either inside the other detection within a distance threshold of 3 or intersect with IOU larger than 0.3, the detection with the highest detection score is kept.
- If an object is not immediately tracked, it will be removed from tracks.
- Objects remain inactive for 30 frames, and then they will be killed. The memory of features of the killed tracks will be released.
- New objects are added to the track, but they can be removed if they are not tracked in the next frame. Objects that are removed by NMS on fully occluded detections also do not start a new track. Previously, I only added high-quality detections as in ByteTrack. But it was not necessary, since the use of NMS.
- In the second stage, a weighted sum of the cosine similarity loss and IOU loss is used for the usual Hungarian matching algorithm, with 0.9 and 0.1 weights, respectively.


Stereo Fish Tracking (SFT)
====================
SFT matches tracks of stereo views. SFT requires videos and tracks in MOT format from each video, along with intrinsic and stereo parameters.

Due to tracking errors, ID switches cause a track to consist of different tracks. This makes stereo matching of tracks challenging.

To tackle the ID switch issue, each track is divided into tracklets, and tracklets are matched based on a (3D) metric. After that, the tracklets are converted into longer tracks based on certain criteria.

SFT has three main steps: track postprocessing, tracklet matching, and making tracks longer. Here is the description of each step:

- Postprocessing tracks: The track postprocessing consists of a few simple steps: remove static tracks, remove short tracks, reindex tracks, and interpolate tracks when frames are missing.
- Tracklet matching: Divide each track into tracklets and then match tracklets based on a metric. The matching is based on the Hungarian method, which is an n-n linear assignment method. A metric is calculated for each pair of tracklets, and the Hungarian method finds the matching pairs. A 3D geometry metric, the sum of differences of the y-component of the rectified coordinates, is used. N.B. We tested different 2D, 3D geometry, and appearance metrics, such as cosine similarity of the deep features for appearance metric, similar 2D shape of the trajectory as a 2D geometry, and 3D coordinates, speed, and acceleration consistency for 3D geometry. We found that the current metric is strong and simple as well. The experiments on different metrics can be seen in `exps/stereo_deep_features.py`.
- Making tracks longer: The tracks become longer if they have the same matching track IDs. If there is only one track ID in common, the disparity continuity, which checks that the difference of disparities does not exceed a threshold, is used. In the case of not having any common track IDs, the track can only become longer if there is only one stereo tracklet holding the disparity continuity. N.B. Due to the complex motion of fishes, the disparity continuity can be easily maintained for wrong tracklet matches.

Either run the below script or use `notebooks/stereo_track_fishes.ipynb`.
```python
scripts/stereo_track_fishes.py configs/stereo_track.yaml
```

Stereo matchting on ground truth data
================
Match the corresponding tracks in stereo tracks. The tracks are matched based on a linear assigment (Hungarian method) of a 3D geometry metric. The 3D geometry metric is simply a mean absolute error in the normalized rectification error (y-axis). The error is normalized by the number of track points. 


Stereo matchting on ground truth data (old)
================
Per track, compute the mean alignment errors and the minimum alignment error is the match. The stereo images should be rectified before.
Requirements: 
- Tracks should be provided.
- rectified images 
- The best works on ground truth tracks mainly: If there are too many short tracklets due to id switch, the method can only match to the best and ignore the rest.  

Track statistics
===========
- tp, fp, fn: number of true positive, false positive, false negative regardless of their track ids.
- no_switch_ids: Total number of times the track id was not correct
- no_unique_ids: number of unique ids. no_unique_ids-1 is implicitly says how many times switch occurs.

#### Correction effort metric:
In this metric, the goal is to measure the amount of effort to correct the mistakes. There are two factors important:
1. If the switch occurs, there is two times cut and merges of tracks. So in total for that switch, there are 4 operations. 
2. Length divided by length of the keyframe.

MOT data of fishes
==================
The data is generated in `exps/fish_tracks.py`.

Association learning
==================

The training is in `scripts/training_al.py`, where dataset, model and training and validation step is in `tracking/association_learning.py`. Some statistics and inference is in `exps\inference_al.py`. Association learning is naively integrated in tracking in `tracking/data_association.py::association_learning_matching`. But this implementation has few bugs mentioned in the text. The main bug is caused by asociating one detection to multiple tracks. The other bug is that time and image is not correct when the last detection of the track is not from the previous frame. 

Data is generated in `exps/data_al_v1.py` and statistics are explained in `data_and_exps.md`. Per (query) detection, 4 frames before and 4 after with positional jitter are cropped to 256x512.  

The association learning still requirs some improvement. Here is the list:
- change the loss from cross entropy to binary cross entropy to make the problem simpler.
- use sin encoding for time (similar as positional encoding from attention is all you need paper.)
- increase the capcity of concatanation encoding from 150 to 2024 or higher. 
- add extra class for unmatched items. This requires to add data without matches. 
- [Most likely not]: aggregate multiple frames (4 previous and 4 after), aggregate their flows. Better temporal encoding should be sufficient, since this step is expensive. 

Postprocessing tracks
==================

The track postprocessing consists of a few simple steps: remove static tracks, remove short tracks, reindex tracks, and interpolate tracks when frames are missing.