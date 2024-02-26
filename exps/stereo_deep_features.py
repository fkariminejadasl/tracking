from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat

from tracking import data_association as da
from tracking import multi_stage_tracking as ms
from tracking import visualize as tv

"""
The bbox embeddings were not descriptive enough.  
"""
"""
gt_matches = {3: 0, 5: 1, 4: 2, 2: 3, 8: 4, 0: 5, 1: 6, 6: 7, 7: 8}

kwargs = ms.get_model_args()

vc1 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
_, image1 = vc1.read()
tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]
det_ids1 = dets1[:, 0]
features1 = ms.calculate_deep_features(det_ids1, dets1, image1, **kwargs)

vc2 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_2.mp4")
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

tv.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1);plt.show(block=False)
tv.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2);plt.show(block=False)
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

vc1 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
_, image1 = vc1.read()
tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]

vc2 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_2.mp4")
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
_, image2 = vc2.read()
tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
dets2 = tracks2[tracks2[:, 1] == 0]

tv.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1)
tv.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2)

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

"""
import numpy as np
import cv2

# Approximate initial values
focal_length = 800  # Example focal length
center = (320, 240)  # Assume the center of the image is the principal point
cameraMatrixL = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype=np.float32)
cameraMatrixR = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype=np.float32)
distCoeffsL = np.zeros((4, 1))  # Assuming no distortion initially
distCoeffsR = np.zeros((4, 1))

# Corresponding image points: imagePointsL and imagePointsR
# imagePointsL = np.array([...]) # Corresponding points in the left image as (N, 2) array
# imagePointsR = np.array([...]) # Corresponding points in the right image as (N, 2) array

# Find the Fundamental Matrix
F, mask = cv2.findFundamentalMat(imagePointsL, imagePointsR, cv2.FM_RANSAC)

# Calculate the Essential Matrix (if intrinsics are known)
E = cameraMatrixR.T @ F @ cameraMatrixL

# Decompose the Essential Matrix to obtain R and T
_, R, T, mask = cv2.recoverPose(E, imagePointsL, imagePointsR, cameraMatrixL, cameraMatrixR)


# Projection matrices
P1 = np.dot(cameraMatrixL, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(cameraMatrixR, np.hstack((R, T.reshape(3, 1))))

# Undistort points
undistortedPointsL = cv2.undistortPoints(np.expand_dims(imagePointsL, axis=1), cameraMatrixL, distCoeffsL, P=P1)
undistortedPointsR = cv2.undistortPoints(np.expand_dims(imagePointsR, axis=1), cameraMatrixR, distCoeffsR, P=P2)

# Triangulate points
points_4D = cv2.triangulatePoints(P1, P2, undistortedPointsL, undistortedPointsR)

# Convert from homogeneous to 3D coordinates
points_3D = points_4D[:3] / points_4D[3]

# points_3D now contains the (X, Y, Z) coordinates of the points in the scene
"""

"""
import numpy as np
import cv2

def reconstruct_3d_points(pts1, pts2, P1, P2):
    # Placeholder for 3D points
    points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    # Convert from homogeneous coordinates to 3D coordinates
    points_3d /= points_3d[3]
    return points_3d[:3]

# Example corresponding points (in pixels)
pts1 = np.float32([[100, 100]]).T
pts2 = np.float32([[110, 100]]).T

# Camera matrices (Identity for this example)
K = np.eye(3)

# Projection matrices for both cameras
P1 = np.hstack((K, np.zeros((3, 1))))  # Camera 1 matrix
t = np.array([[1, 0, 0]]).T  # Translation vector for Camera 2
P2 = np.hstack((K, t))  # Camera 2 matrix with translation

# Triangulate points
points_3d = reconstruct_3d_points(pts1, pts2, P1, P2)

print("3D Points:\n", points_3d)
"""


gt_matches = {3: 0, 5: 1, 4: 2, 2: 3, 8: 4, 0: 5, 1: 6, 6: 7, 7: 8}
dd = loadmat("/home/fatemeh/Downloads/fish/mot_data//stereo_129.mat")
image1 = cv2.imread("/home/fatemeh/Downloads/fish/mot_data/129_1_0.png")
image2 = cv2.imread("/home/fatemeh/Downloads/fish/mot_data/129_2_0.png")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
dets2 = tracks2[tracks2[:, 1] == 0]

a = []
for id in gt_matches.keys():
    a.append(dets1[id])
a = np.array(a)
dets1 = a

distCoeffs1 = deepcopy(dd["distortionCoefficients1"])
distCoeffs2 = deepcopy(dd["distortionCoefficients2"])
cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"])
cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"])
R = deepcopy(dd["rotationOfCamera2"])
T = deepcopy(dd["translationOfCamera2"])
imagePoints1 = dets1[:, 7:9].astype(np.float32)
imagePoints2 = dets2[:, 7:9].astype(np.float32)

cameraMatrix1[0:2, 2] += 1
cameraMatrix2[0:2, 2] += 1

# Projection matrices
P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(cameraMatrix2, np.hstack((R, T.reshape(3, 1))))

# Undistort points
undistortedPoints1 = cv2.undistortPoints(
    imagePoints1, cameraMatrix1, distCoeffs1, P=cameraMatrix1
)  # result Nx1x2. No need for np.expand_dims(imagePoints1, axis=1)
undistortedPoints2 = cv2.undistortPoints(
    imagePoints2, cameraMatrix2, distCoeffs2, P=cameraMatrix2
)

points_4D = cv2.triangulatePoints(P1, P2, undistortedPoints1, undistortedPoints2)

# Convert from homogeneous to 3D coordinates
points_3D = points_4D[:3] / points_4D[3]

print(points_3D.T)
print(dets1)
print("end")


"""Matlab
# I had to check Matlab because my python result was not correct. Issue due to:
# 1. undistortPoints: P should be camera matrix (main issue)
# 2. conversion of matlab to opencv parameters shifting principal point to be 0-based
# was gave larger error in order of mm. Now error is order of .01 mm. (sub issue)
# I put the matlab code here.
load('/home/fatemeh/Downloads/fish/mot_data/stereo_129.mat')
image1 = imread("/home/fatemeh/Downloads/fish/mot_data/129_1_0.png");
image2 = imread("/home/fatemeh/Downloads/fish/mot_data/129_2_0.png");
matchedPoints1 = [1088,  340; 1224,  323; 1231,  330; 1495,  441; 1206,  595; 1225,  423; 1164,  419; 1079,  504; 1099,  479];
matchedPoints2 = [1020,  360; 1161,  346; 1178,  346; 1455,  470; 1090,  629; 1116,  447; 1055,  444;  996,  532;  992,  504];
undistortedPoints1 = undistortPoints(matchedPoints1, stereoParams.CameraParameters1);
undistortedPoints2 = undistortPoints(matchedPoints2, stereoParams.CameraParameters2);
load('/home/fatemeh/Downloads/fish/mot_data/stereo_129.mat');
worldPoints = triangulate(undistortedPoints1, undistortedPoints2, stereoParams);
[image1Rect, image2Rect] = rectifyStereoImages(image1, image2, stereoParams);
% figure;imshow(image1);title("image1")
% figure;imshow(image2);title("image2")
% figure;imshow(image1Rect);title("image1Rect")
% figure;imshow(image2Rect);title("image2Rect")
gray1 = rgb2gray(image1);
gray2 = rgb2gray(image2);
rgray1 = rgb2gray(image1Rect);
rgray2 = rgb2gray(image2Rect);
figure();imshow(stereoAnaglyph(gray1, gray2))
figure();imshow(stereoAnaglyph(rgray1, rgray2))
showExtrinsics(stereoParams)
"""