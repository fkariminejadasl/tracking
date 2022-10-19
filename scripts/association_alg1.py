from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

result_folder = Path("/home/fatemeh/results/dataset1")
data_folder = Path("/home/fatemeh/data/dataset1")
track_folder = (
    data_folder / "cam1_labels/cam1_labels"
)  # cam1_labels/cam1_labels, cam2_labels/cam2_labels
filename_fixpart = "12_07_22_1_C_GH040468_1_cam1_rect"  # 12_07_22_1_C_GH040468_1_cam1_rect, 12_07_22_1_D_GH040468_1_cam2_rect
vc = cv2.VideoCapture((data_folder / f"{filename_fixpart}.mp4").as_posix())
vc.isOpened()


# visualize detection as video
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vc.get(cv2.CAP_PROP_FPS)


def match_two_detection_sets(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            dist[i, j] = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def draw_matches(frame1, frame2, matches1, matches2):
    _, ax1 = plt.subplots(1, 1)
    _, ax2 = plt.subplots(1, 1)
    ax1.imshow(frame1[..., ::-1])
    ax2.imshow(frame2[..., ::-1])
    ax1.set_xlim(1400, 2100)
    ax1.set_ylim(1200, 700)
    ax2.set_xlim(1400, 2100)
    ax2.set_ylim(1200, 700)

    for match1, match2 in zip(matches1, matches2):
        ax1.plot([match1.x, match2.x], [match1.y, match2.y], "*-", color=(0, 0, 1))
        ax2.plot([match1.x, match2.x], [match1.y, match2.y], "*-", color=(0, 0, 1))
    plt.show(block=False)


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    id: int


@dataclass
class Track:
    coords: list[Detection]
    color: tuple
    frames: list[int]


def get_detections(det_path) -> list[Detection]:
    detections = np.loadtxt(det_path)
    return [
        Detection(
            x=int(det[1] * width),
            y=int(det[2] * height),
            w=int(det[3] * width),
            h=int(det[4] * height),
            id=i,
        )
        for i, det in enumerate(detections)
    ]


# initiate track
det_path1 = track_folder / f"{filename_fixpart}_{1}.txt"
det_path2 = track_folder / f"{filename_fixpart}_{2}.txt"
dets1 = get_detections(det_path1)
dets2 = get_detections(det_path2)
ids1, ids2 = match_two_detection_sets(dets1, dets2)
tracks = {}
for id1, id2 in zip(ids1, ids2):
    coords = [dets1[id1], dets2[id2]]
    color = tuple(np.random.rand(3))
    frames = [1, 2]
    track = Track(coords=coords, color=color, frames=frames)
    tracks[id1] = track

for frame_number in range(2, 10):
    det_path2 = track_folder / f"{filename_fixpart}_{frame_number}.txt"
    det_path3 = track_folder / f"{filename_fixpart}_{frame_number+1}.txt"
    dets2 = get_detections(det_path2)
    dets3 = get_detections(det_path3)
    ids1, ids2 = match_two_detection_sets(dets1, dets2)
    ids2_2, ids3 = match_two_detection_sets(dets2, dets3)

    id2_id1 = {id2: id1 for id1, id2 in zip(ids1, ids2)}
    common_ids2 = set(ids2).intersection(ids2_2)
    id2_2_id3 = {id2_2: id3 for id2_2, id3 in zip(ids2_2, ids3)}
    for common_id2 in common_ids2:
        id1 = id2_id1[common_id2]
        id3 = id2_2_id3[common_id2]
        # remove wrong matches
        det_new = dets3[id3]
        det_previous = tracks[id1].coords[-1]
        dist = np.linalg.norm([det_new.x - det_previous.x, det_new.y - det_previous.y])
        if dist < 10:  # TODO fedge factor
            tracks[id1].coords.append(dets3[id3])
            tracks[id1].frames.append(frame_number + 1)

# visualize tracks
vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, frame1 = vc.read()
plt.figure()
plt.imshow(frame1[..., ::-1])
for _, track in tracks.items():
    plt.plot(
        [det.x for det in track.coords],
        [det.y for det in track.coords],
        "*-",
        color=track.color,
    )
plt.show(block=False)
