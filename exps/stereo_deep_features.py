from pathlib import Path

import cv2
import numpy as np

from tracking import data_association as da
from tracking import multi_stage_tracking as ms
from tracking import visualize

"""
The bbox embeddings were not descriptive enough.  
"""

gt_matches = {2: 3, 0: 5, 1: 6, 5: 1, 4: 2, 3: 0, 7: 8, 6: 7, 4: 4}

kwargs = ms.get_model_args()

vc1 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids_tracks/129_1.mp4")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
_, image1 = vc1.read()
tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]
det_ids1 = dets1[:, 0]
features1 = ms.calculate_deep_features(det_ids1, dets1, image1, **kwargs)

vc2 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids_tracks/129_2.mp4")
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
_, image2 = vc2.read()
tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
dets2 = tracks2[tracks2[:, 1] == 0]
det_ids2 = dets2[:, 0]
features2 = ms.calculate_deep_features(det_ids2, dets2, image2, **kwargs)


csims = dict()
for did1, f1 in features1.items():
    for did2, f2 in features2.items():
        csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        csims[(did1, did2)] = csim

[csims[(0, i)] for i in det_ids2]

visualize.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1)
visualize.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2)
