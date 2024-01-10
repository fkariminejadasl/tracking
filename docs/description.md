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


Stereo matchting on ground truth data
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

