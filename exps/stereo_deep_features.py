from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking import data_association as da
from tracking import multi_stage_tracking as ms
from tracking import visualize as tv

"""
The bbox embeddings 
"""
"""
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}

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
dist = np.zeros((len(features1), len(features2)), dtype=np.float32)
for did1, f1 in features1.items():
    for did2, f2 in features2.items():
        csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        csims[(did1, did2)] = csim
        dist[did1, did2] = 1 - csim

[csims[(0, i)] for i in det_ids2]

tv.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1);plt.show(block=False)
tv.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2);plt.show(block=False)
rows, cols = linear_sum_assignment(dist)
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

image0 = load_image("/home/fatemeh/Downloads/fish/mot_data/left.png")
image1 = load_image("/home/fatemeh/Downloads/fish/mot_data/right.png")

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
# One image 3D position, rectified coordinate and rectified images
# =========
gt_matches = {3: 0, 5: 1, 4: 2, 2: 3, 8: 4, 0: 5, 1: 6, 6: 7, 7: 8}
dd = loadmat("/home/fatemeh/Downloads/fish/mot_data//stereo_129.mat")
image1 = cv2.imread("/home/fatemeh/Downloads/fish/mot_data/129_1_0.png")
image2 = cv2.imread("/home/fatemeh/Downloads/fish/mot_data/129_2_0.png")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)

distCoeffs1 = deepcopy(dd["distortionCoefficients1"])[0].astype(np.float64)
distCoeffs2 = deepcopy(dd["distortionCoefficients2"])[0].astype(np.float64)
cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"]).astype(np.float64)
cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"]).astype(np.float64)
R = deepcopy(dd["rotationOfCamera2"]).astype(np.float64)
T = deepcopy(dd["translationOfCamera2"]).T.astype(np.float64)
cameraMatrix1[0:2, 2] += 1
cameraMatrix2[0:2, 2] += 1
# Projection matrices
P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
P2 = np.dot(cameraMatrix2, np.hstack((R, T))).astype(np.float64)

tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
dets1 = tracks1[tracks1[:, 1] == 0]
tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
dets2 = tracks2[tracks2[:, 1] == 0]

a = []
for id in gt_matches.keys():
    a.append(dets1[id])
a = np.array(a)
dets1 = a

imagePoints1 = dets1[:, 7:9].astype(np.float32)
imagePoints2 = dets2[:, 7:9].astype(np.float32)

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

worldPoints = 1e3 * np.array(
    [
        [0.2928, -0.4759, 2.6248],
        [0.5945, -0.5079, 2.5994],
        [0.6387, -0.5236, 2.7220],
        [1.2068, -0.2507, 2.5901],
        [0.4596, 0.0793, 2.1652],
        [0.4974, -0.2415, 2.1729],
        [0.3884, -0.2499, 2.2020],
        [0.2599, -0.1001, 2.5094],
        [0.2724, -0.1413, 2.2591],
    ]
)
print(worldPoints - points_3D.T)

image_size = image1.shape[1], image1.shape[0]
R1, R2, r_P1, r_P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
    imageSize=image_size,
    R=R,
    T=T,
)  # , flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
map1x, map1y = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, r_P1, image_size, cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, r_P2, image_size, cv2.CV_32FC1
)
rectified_im1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
rectified_im2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)
# recotify points
rectified_points1 = cv2.undistortPoints(
    imagePoints1, cameraMatrix1, distCoeffs1, R=R1, P=r_P1
).squeeze(axis=1)
rectified_points2 = cv2.undistortPoints(
    imagePoints2, cameraMatrix2, distCoeffs2, R=R2, P=r_P2
).squeeze(axis=1)

print("end")

'''Matlab
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

# I got from matlab
worldPoints = 1e3 * np.array([
[0.2928,-0.4759,2.6248],
[0.5945,-0.5079,2.5994],
[0.6387,-0.5236,2.7220],
[1.2068,-0.2507,2.5901],
[0.4596, 0.0793,2.1652],
[0.4974,-0.2415,2.1729],
[0.3884,-0.2499,2.2020],
[0.2599,-0.1001,2.5094],
[0.2724,-0.1413,2.2591]])
'''
"""


# fmt: off
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# gt_matches = {3:0, 5:0, 4:0, 2:0, 8:0, 0:0, 1:0, 6:0, 7:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
# fmt: on
dd = loadmat("/home/fatemeh/Downloads/fish/mot_data//stereo_129.mat")
vc1 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_1.mp4")
vc2 = cv2.VideoCapture("/home/fatemeh/Downloads/fish/mot_data/vids/129_2.mp4")
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)

max_frames = min(
    int(vc1.get(cv2.CAP_PROP_FRAME_COUNT)), int(vc2.get(cv2.CAP_PROP_FRAME_COUNT))
)
step = 1

distCoeffs1 = deepcopy(dd["distortionCoefficients1"])[0].astype(np.float64)
distCoeffs2 = deepcopy(dd["distortionCoefficients2"])[0].astype(np.float64)
cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"]).astype(np.float64)
cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"]).astype(np.float64)
R = deepcopy(dd["rotationOfCamera2"]).astype(np.float64)
T = deepcopy(dd["translationOfCamera2"]).T.astype(np.float64)
cameraMatrix1[0:2, 2] += 1
cameraMatrix2[0:2, 2] += 1
# Projection matrices
P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
P2 = np.dot(cameraMatrix2, np.hstack((R, T))).astype(np.float64)


match_frames = np.empty((0, 7))
for frame_number in tqdm(range(0, max_frames, step)):
    _, image1 = vc1.read()
    _, image2 = vc2.read()
    if frame_number % step != 0:
        continue

    tracks1[:, 2] = tracks1[:, 0]  # dets or tracks used as dets
    dets1 = tracks1[tracks1[:, 1] == frame_number]
    tracks2[:, 2] = tracks2[:, 0]  # dets or tracks used as dets
    dets2 = tracks2[tracks2[:, 1] == frame_number]

    n_dets1 = []
    n_dets2 = []
    for id1, id2 in gt_matches.items():
        if (id1 in dets1[:, 0]) and (id2 in dets2[:, 0]):
            det1 = dets1[dets1[:, 0] == id1][0]
            det2 = dets2[dets2[:, 0] == id2][0]
            n_dets1.append(det1)
            n_dets2.append(det2)
    dets1 = np.array(n_dets1)
    dets2 = np.array(n_dets2)

    imagePoints1 = dets1[:, 7:9].astype(np.float32)
    imagePoints2 = dets2[:, 7:9].astype(np.float32)

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

    extras = np.stack(
        (dets2[:, 0], frame_number * np.ones(len(dets1)), dets1[:, 0], dets2[:, 0])
    ).T
    match_frame = np.concatenate((extras, points_3D.T), axis=1)
    match_frames = np.concatenate((match_frames, match_frame), axis=0)


def grad_3d(coor):
    dcoor = np.stack(
        (
            coor[1:, 0],
            np.diff(coor[:, 1]) * 1e-3 / (np.diff(coor[:, 0]) / 240),
            np.diff(coor[:, 2]) * 1e-3 / (np.diff(coor[:, 0]) / 240),
            np.diff(coor[:, 3]) * 1e-3 / (np.diff(coor[:, 0]) / 240),
        )
    ).T
    return dcoor


def smooth_3d(coor, sigma):
    x = gaussian_filter(coor[:, 1], sigma=sigma)
    y = gaussian_filter(coor[:, 2], sigma=sigma)
    z = gaussian_filter(coor[:, 3], sigma=sigma)
    smoothed_coor = np.stack((coor[:, 0], x, y, z)).T
    return smoothed_coor


def calculate_angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between two vectors."""
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) / np.pi * 180  # Convert to degrees
    return angle


coor = match_frames[match_frames[:, 2] == 3]
coor = coor[:, [1, 4, 5, 6]]
s_coor = smooth_3d(coor, 16)


# interactive visualization of 3D points with time as slider
# =========
from matplotlib.widgets import Slider

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initial plot with the full trajectory for context
ax.plot(coor[:, 1], coor[:, 2], coor[:, 3], color="gray", alpha=0.5)

# Plot the initial state
(line,) = ax.plot(coor[0:1, 1], coor[0:1, 2], coor[0:1, 3], "*-")

# [left, bottom, width, height]
axtime = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
stime = Slider(
    axtime, "Time", coor[0, 0], coor[-1, 0].max(), valinit=coor[0, 0], valstep=1
)


def update(val):
    time = stime.val
    updated_coor = coor[coor[:, 0] <= time]
    line.set_data(updated_coor[:, 1], updated_coor[:, 2])
    line.set_3d_properties(updated_coor[:, 3])
    fig.canvas.draw_idle()


stime.on_changed(update)
# ======


dcoor = grad_3d(s_coor)
ddcoor = grad_3d(dcoor)

angles = []
for i in range(0, len(dcoor) - 1):
    angle = calculate_angle_between_vectors(dcoor[i, 1:], dcoor[i + 1, 1:])
    angles.append(angle)


angles2d = []
for i in range(0, len(dcoor) - 1):
    angle = calculate_angle_between_vectors(dcoor[i, 1:3], dcoor[i + 1, 1:3])
    angles2d.append(angle)

plt.figure()
ax = plt.subplot(projection="3d")
ax.plot(s_coor[:, 1], s_coor[:, 2], s_coor[:, 3], "*-")
plt.figure()
plt.plot(angles, "-*")
print("end")


# gt_matches = {3:0, 5:0, 4:0, 2:0, 8:0, 0:0, 1:0, 6:0, 7:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
for i in range(0, 15):
    coor = match_frames[match_frames[:, 2] == i]
    coor = coor[:, [1, 4, 5, 6]]
    dcoor = grad_3d(coor)
    speed_mag = np.linalg.norm(dcoor[:, 1:], axis=1)

    # fmt: off
    # depth
    plt.figure();plt.plot(coor[:, 0], coor[:, -1], "*-");plt.title(f"depth: {i}")
    plt.figure();plt.plot(dcoor[:, 0], dcoor[:, -1], "*-");plt.title(f"ddepth: {i}")

    # speed
    plt.figure();plt.plot(dcoor[:, 0], speed_mag, "*-");plt.title(f"speed mag: {i}")
    plt.figure();plt.hist(speed_mag);plt.title(f"speed mag: {i}")
    plt.figure();ax = plt.subplot(projection="3d");ax.plot(coor[:, 1], coor[:, 2], coor[:, 3], "*-");plt.title(f"{i}")
    # fmt: on

    # s_coor = smooth_3d(coor, 16)
    # plt.figure();plt.plot(s_coor[:,0], s_coor[:,3],"*-");plt.title(f"{i}")
    # plt.figure()
    # ax = plt.subplot(projection="3d")
    # ax.plot(s_coor[:, 1], s_coor[:, 2], s_coor[:, 3], "*-")
    # plt.title(f"{i}")

print("end")

# TODO: remove
for i in range(0, 15):
    coor = match_frames[match_frames[:, 2] == i]
    coor = coor[:, [1, 4, 5, 6]]
    dcoor = np.diff(coor, axis=0)
    speed_mag = np.linalg.norm(dcoor[:, 1:], axis=1)
    # speed
    plt.figure()
    plt.plot(coor[1:, 0], speed_mag, "*-")
    plt.title(f"speed mag: {i}")
# TODO: remove

"""
# Test parameters are equal (129 is the dept4)
st_path = Path("/home/fatemeh/Downloads/fish/mot_data/stereo_parameters")
for i in range(1, 13):
    print(i)
    dd1 = loadmat(st_path/f"stereoParams_Dep{i}.mat")
    dd2 = loadmat(st_path/f"stereoParams_Dep{i}_OpenCV.mat")
    np.testing.assert_allclose(dd1["distortionCoefficients1"], dd2["distortionCoefficients1"])
    np.testing.assert_allclose(dd1["distortionCoefficients2"], dd2["distortionCoefficients2"])
    np.testing.assert_allclose(dd1["intrinsicMatrix1"], dd2["intrinsicMatrix1"])
    np.testing.assert_allclose(dd1["intrinsicMatrix2"], dd2["intrinsicMatrix2"])
    np.testing.assert_allclose(dd1["rotationOfCamera2"], dd2["rotationOfCamera2"])
    np.testing.assert_allclose(dd1["translationOfCamera2"], dd2["translationOfCamera2"])
"""
