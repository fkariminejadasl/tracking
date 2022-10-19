from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

data_folder = Path("/home/fatemeh/data/dataset1")
result_folder = Path("/home/fatemeh/results/dataset1")
track_folder1 = Path("/home/fatemeh/data/dataset1/cam1_labels/cam1_labels")
track_folder2 = Path("/home/fatemeh/data/dataset1/cam2_labels/cam2_labels")
vc1 = cv2.VideoCapture(
    (data_folder / "12_07_22_1_C_GH040468_1_cam1_rect 1.mp4").as_posix()
)
vc2 = cv2.VideoCapture(
    (data_folder / "12_07_22_1_D_GH040468_1_cam2_rect 1.mp4").as_posix()
)


# visualize detection as video
height = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
total_no_frames = int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vc1.get(cv2.CAP_PROP_FPS)


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    id: int


def get_detections(det_path) -> list[Detection]:
    detections = np.loadtxt(det_path)
    return [
        Detection(
            x = int(det[1] * width),
            y = int(det[2] * height),
            w = int(det[3] * width),
            h = int(det[4] * height),
            id = i,
        )
        for i, det in enumerate(detections)
    ]


vc1.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, frame1_1 = vc1.read()
vc2.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, frame1_2 = vc2.read()


det_path1 = track_folder1 / f"12_07_22_1_C_GH040468_1_cam1_rect_{1}.txt"
det_path2 = track_folder2 / f"12_07_22_1_D_GH040468_1_cam2_rect_{1}.txt"
dets1 = get_detections(det_path1)
dets2 = get_detections(det_path2)


def draw_detections(frame, dets):
    for det in dets:
        w2 = int(det.w / 2)
        h2 = int(det.h / 2)
        color = tuple(int(i) for i in np.random.randint(0, 255, (3,)))
        cv2.rectangle(
            frame,
            (det.x - w2, det.y - h2),
            (det.x + w2, det.y + h2),
            color=color,
            thickness=1,
        )
    plt.figure()
    plt.imshow(frame[..., ::-1])
    plt.show(block=False)


# cluster detections with similar y-coodinates
# similarity threshold = 10
@dataclass
class Cluster:
    ids1: list[int]
    ids2: list[int]


def cluster_by_y(dets1, sim_thres=10):
    data = np.array([(0.0, float(det1.y)) for det1 in dets1])
    ids1 = fcluster(ward(pdist(data)), t=sim_thres, criterion="distance")
    c1 = {}
    for i, id1 in enumerate(ids1):
        if id1 in c1:
            c1[id1].append(i)
        else:
            c1[id1] = [i]
    return c1

cluster1 = cluster_by_y(dets1) 
cluster2 = {}
for det1 in dets1:
    for det2 in dets2:
        if (abs(det1.y-det2.y)<10) and (det1.id is not det2.id):
            if det1.id in cluster2:
                cluster2[det1.id].append(det2.id)
            else:
                cluster2[det1.id] = [det2.id]

cluster = {}
for i, (key1, values1) in enumerate(cluster1.items()):
    ids2 = set()
    for value1 in values1:
        if value1 in cluster2:
            ids2 = ids2.union(set(cluster2[value1]))
    cluster[i] = Cluster(ids1=values1, ids2=list(ids2))


# TODO
# problem is that that cluster1 is good but cluster2 is bad. distance become larger
# maybe I can do Hungerian without clustering only on flows. 

# currently:
# cluster on similar y
# for each cluster find similar flow


