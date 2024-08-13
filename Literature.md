Multi Object Tracking
====================

## End-to-End Tracking

**MOTR Family**

- [MeMOTR](https://arxiv.org/abs/2307.15700)
- [MOTRv3](https://arxiv.org/abs/2305.14298) and [CO-MOT](https://arxiv.org/abs/2305.12724): Improvement on [MOTR](https://arxiv.org/abs/2105.03247) by increasing detection objects in the loss term as extra supervision.
- [MO-YOLO](https://arxiv.org/abs/2310.17170): : Efficient (fast training on 1 2080 ti GPU, 8 hours) YOLO with transformer in the encoder (RT-YOLO) and MOTR in the decoder.


Stereo Tracking
===============
- My resnet deep features (exp/stereo_deep_features.py)
- LightGLUE with a hack to get bounding boxes: I think stronger that than LightGLUE is [CoTracker](https://arxiv.org/abs/2307.07635) from the family of point tracker or TAP (track any point), which take both time and space into account. But [CoTracker](https://arxiv.org/abs/2307.07635) is only in limited to track in video not in a multi-view setup. 
- [Match Anything](https://arxiv.org/abs/2406.04221): It is a strong representation learning by using SAM and strong data augmentation, learns the similarity. This is less suitable for the fish data I guess.
- [MMCT](https://arxiv.org/abs/2312.11035) combines a tracklet based on a simple only trajectories plus frame number