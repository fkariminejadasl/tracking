import enum
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

accepted_flow_length = 10

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


class Status(enum.Enum):
    Tracked: bool = 1
    Untracked: bool = 2
    Stoped: bool = 3


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    frameid: int


@dataclass
class Track:
    coords: list[Detection]
    predicted_loc: Detection
    color: tuple
    frameids: list[int]
    status: Status


def get_detections(det_path, frame_number) -> list[Detection]:
    detections = np.loadtxt(det_path)
    return [
        Detection(
            x=int(det[1] * width),
            y=int(det[2] * height),
            w=int(det[3] * width),
            h=int(det[4] * height),
            frameid=frame_number,
        )
        for det in detections
    ]


def match_two_detection_sets(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            dist[i, j] = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


# initiate track
frame_number1 = 1
frame_number2 = 10
det_path1 = track_folder / f"{filename_fixpart}_{frame_number1}.txt"
det_path2 = track_folder / f"{filename_fixpart}_{frame_number2}.txt"
dets1 = get_detections(det_path1, frame_number1)
dets2 = get_detections(det_path2, frame_number2)
ids1, ids2 = match_two_detection_sets(dets1, dets2)
tracks = {}
# matched tracks
track_id = 0
flows_x = []
flows_y = []
for id1, id2 in zip(ids1, ids2):
    coords = [dets1[id1], dets2[id2]]
    color = tuple(np.random.rand(3))
    frameids = [frame_number1, frame_number2]
    flow_x = dets2[id2].x - dets1[id1].x
    flow_y = dets2[id2].y - dets1[id1].y
    pred_x = flow_x + dets2[id2].x
    pred_y = flow_y + dets2[id2].y
    flow_length = np.linalg.norm([flow_x, flow_y])
    if flow_length < accepted_flow_length:
        flows_x.append(flow_x)
        flows_y.append(flow_y)
    predicted_loc = Detection(
        x=pred_x,
        y=pred_y,
        w=dets2[id2].w,
        h=dets2[id2].h,
        frameid=dets2[id2].frameid + 1,
    )
    track = Track(
        coords=coords,
        predicted_loc=predicted_loc,
        color=color,
        frameids=frameids,
        status=Status.Tracked,
    )
    tracks[track_id] = track
    track_id += 1
common_flow_x = np.median(np.array(flows_x))
common_flow_y = np.median(np.array(flows_y))
# unmatched tracks: frame1
diff_ids = set(range(len(dets1))).difference(set(ids1))
for id in diff_ids:
    coords = [dets1[id]]
    frameids = [frame_number1]
    flow_x = common_flow_x * 2
    flow_y = common_flow_y * 2
    pred_frameid = frame_number1 + 2
    pred_x = flow_x + coords[0].x
    pred_y = flow_y + coords[0].y
    color = tuple(np.random.rand(3))
    predicted_loc = Detection(
        x=pred_x, y=pred_y, w=coords[0].w, h=coords[0].h, frameid=pred_frameid
    )
    track = Track(
        coords=coords,
        predicted_loc=predicted_loc,
        color=color,
        frameids=frameids,
        status=Status.Untracked,
    )
    tracks[track_id] = track
    track_id += 1
# unmatched tracks: frame2
diff_ids = set(range(len(dets2))).difference(set(ids2))
for id in diff_ids:
    coords = [dets2[id]]
    frameids = [frame_number2]
    flow_x = common_flow_x
    flow_y = common_flow_y
    pred_frameid = frame_number2 + 1
    pred_x = flow_x + coords[0].x
    pred_y = flow_y + coords[0].y
    color = tuple(np.random.rand(3))
    predicted_loc = Detection(
        x=pred_x, y=pred_y, w=coords[0].w, h=coords[0].h, frameid=pred_frameid
    )
    track = Track(
        coords=coords,
        predicted_loc=predicted_loc,
        color=color,
        frameids=frameids,
        status=Status.Untracked,
    )
    tracks[track_id] = track
    track_id += 1

"""
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
"""

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


# visualize tracks on two frames
vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number1 - 1)
_, frame1 = vc.read()
vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number2 - 1)
_, frame2 = vc.read()
_, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=True);ax1.imshow(frame1[...,::-1]);ax2.imshow(frame2[...,::-1])
for _, track in tracks.items():
    for id in track.frameids:
        if id == frame_number1:
            ax1.plot([track.coords[0].x], [track.coords[0].y], "*", color=track.color)
        if id == frame_number2:
            ax2.plot([track.coords[1].x], [track.coords[1].y], "*", color=track.color)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show(block=False)
