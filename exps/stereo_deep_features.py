from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np

from tracking import data_association as da
from tracking import multi_stage_tracking as ms
from tracking import visualize

"""
The bbox embeddings were not descriptive enough.  
"""
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

visualize.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1);plt.show(block=False)
visualize.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2);plt.show(block=False)
"""

"""
import sys
from pathlib import Path

print(sys.path)

import torch
from lightglue import DISK, SIFT, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

# extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="sift").eval().to(device)  # superpoint
extractor = SIFT(max_num_keypoints=2048).eval().to(device)

image0 = load_image("/home/fatemeh/Downloads/left.png")
image1 = load_image("/home/fatemeh/Downloads/right.png")

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
print("")
"""

"""
import sys
from pathlib import Path

print(sys.path)

import torch
from lightglue import DISK, SIFT, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

vc1 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids_tracks/129_1.mp4")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
_, image1 = vc1.read()
tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]

vc2 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids_tracks/129_2.mp4")
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
_, image2 = vc2.read()
tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
dets2 = tracks2[tracks2[:, 1] == 0]

visualize.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1)
visualize.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2)

gimage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints1 = [
    cv2.KeyPoint(float(det[7]), float(det[8]), float(min(det[9], det[10])))
    for det in dets1
]
keypoints2 = [
    cv2.KeyPoint(float(det[7]), float(det[8]), float(min(det[9], det[10])))
    for det in dets2
]
keypoints1, descriptors1 = sift.compute(gimage1, keypoints1)
keypoints2, descriptors2 = sift.compute(gimage2, keypoints2)
keypoints1, descriptors1 = sift.compute(gimage1, [cv2.KeyPoint(1225, 423, 14)])
# detections, scores, descriptors
# feats0['descriptors'].dtype, feats0['keypoints'].dtype, feats0['scales'].dtype, feats0['image_size'].dtype # float32
# feats0['descriptors'].shape, feats0['keypoints'].shape, feats0['scales'].shape, feats0['image_size'].shape
# (torch.Size([1, 1657, 128]), torch.Size([1, 1657, 2]), torch.Size([1, 1657]), torch.Size([1, 2]))
features1 = dict()
features1["image_size"] = (
    torch.tensor([[image1.shape[1], image1.shape[0]]]).to(device).to(torch.float32)
)
features1["scales"] = torch.tensor([[float(min(det[9], det[10])) for det in dets1]]).to(
    device
)
features1["keypoints"] = torch.tensor(
    [[[float(det[7]), float(det[8])] for det in dets1]]
).to(device)
features1["descriptors"] = torch.tensor(descriptors1).unsqueeze(0).to(device)

# torch.Size([1, 3, 1080, 1920]) # 0.5333333333333333
# torch.Size([1, 3, 576, 1024]) # 0.5333333333333333
# feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
"""
