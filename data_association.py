import enum
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

accepted_flow_length = 10
stopped_track_length = 50

result_folder = Path("/home/fatemeh/results/dataset1")
data_folder = Path("/home/fatemeh/data/dataset1")
det_folder1 = data_folder / "cam1_labels/cam1_labels"
det_folder2 = data_folder / "cam2_labels/cam2_labels"
filename_fixpart1 = "12_07_22_1_C_GH040468_1_cam1_rect"
filename_fixpart2 = "12_07_22_1_D_GH040468_1_cam2_rect"
vc1 = cv2.VideoCapture((data_folder / f"{filename_fixpart1}.mp4").as_posix())
vc2 = cv2.VideoCapture((data_folder / f"{filename_fixpart2}.mp4").as_posix())


def get_video_parameters(vc: cv2.VideoCapture):
    if vc.isOpened():
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vc.get(cv2.CAP_PROP_FPS)
        return height, width, total_no_frames, fps
    else:
        return


height, width, total_no_frames, fps = get_video_parameters(vc1)


class Status(enum.Enum):
    Tracked: bool = enum.auto()
    Untracked: bool = enum.auto()
    Stoped: bool = enum.auto()
    NewTrack: bool = enum.auto()


@dataclass
class Point:
    x: float
    y: float


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
    predicted_loc: Detection
    color: tuple
    frameids: list[int]
    status: Status


def get_detections(det_path, frame_number, width: int, height: int) -> list[Detection]:
    detections = np.loadtxt(det_path)
    return [
        Detection(
            x=int(det[1] * width),
            y=int(det[2] * height),
            w=int(det[3] * width),
            h=int(det[4] * height),
            id=frame_number,
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


def _make_a_new_track(coords, frameids, flow, track_id, status) -> Track:
    color = tuple(np.random.rand(3).astype(np.float16))
    pred = Point(x=flow.x + coords[-1].x, y=flow.y + coords[-1].y)
    predicted_loc = Detection(
        x=pred.x, y=pred.y, w=coords[-1].w, h=coords[-1].h, id=track_id
    )
    track = Track(
        coords=coords,
        predicted_loc=predicted_loc,
        color=color,
        frameids=frameids,
        status=status,
    )
    return track


def _get_common_flow(flows):
    common_flow = Point(
        x=np.average(np.array([flow.x for flow in flows])),
        y=np.average(np.array([flow.y for flow in flows])),
    )
    return common_flow


def compute_tracks(det_folder: Path, filename_fixpart: str, width: int, height: int):
    # initiate track
    # ===============
    frame_number1 = 1
    frame_number2 = 2
    det_path1 = det_folder / f"{filename_fixpart}_{frame_number1}.txt"
    det_path2 = det_folder / f"{filename_fixpart}_{frame_number2}.txt"
    dets1 = get_detections(det_path1, frame_number1, width, height)
    dets2 = get_detections(det_path2, frame_number2, width, height)
    ids1, ids2 = match_two_detection_sets(dets1, dets2)
    tracks = {}
    track_id = 0
    # matched tracks
    flows = [Point(x=0.0, y=0.0)]
    for id1, id2 in zip(ids1, ids2):
        dist = np.linalg.norm(
            [dets1[id1].x - dets2[id2].x, dets1[id1].y - dets2[id2].y]
        )
        if dist < accepted_flow_length:
            coords = [dets1[id1], dets2[id2]]
            frameids = [frame_number1, frame_number2]
            flow = Point(x=coords[-1].x - coords[-2].x, y=coords[-1].y - coords[-2].y)
            flow_length = np.linalg.norm([flow.x, flow.y])
            if flow_length < accepted_flow_length:
                flows.append(flow)
            tracks[track_id] = _make_a_new_track(
                coords, frameids, flow, track_id, Status.Tracked
            )
            track_id += 1
        else:
            common_flow = _get_common_flow(flows)
            tracks[track_id] = _make_a_new_track(
                [dets1[id1]], [frame_number1], common_flow, track_id, Status.NewTrack
            )
            track_id += 1
            tracks[track_id] = _make_a_new_track(
                [dets2[id2]], [frame_number2], common_flow, track_id, Status.NewTrack
            )
            track_id += 1

    common_flow = _get_common_flow(flows)

    # unmatched tracks: frame1
    diff_ids = set(range(len(dets1))).difference(set(ids1))
    for id in diff_ids:
        coords = [dets1[id]]
        frameids = [frame_number1]
        flow = Point(x=common_flow.x * 2, y=common_flow * 2)

        tracks[track_id] = _make_a_new_track(
            coords, frameids, flow, track_id, Status.NewTrack
        )
        track_id += 1
    # unmatched tracks: frame2
    diff_ids = set(range(len(dets2))).difference(set(ids2))
    for id in diff_ids:
        coords = [dets2[id]]
        frameids = [frame_number2]
        flow = common_flow

        tracks[track_id] = _make_a_new_track(
            coords, frameids, flow, track_id, Status.NewTrack
        )
        track_id += 1

    # start track
    # ===========
    for frame_number in range(3, 681):
        det_path = det_folder / f"{filename_fixpart}_{frame_number}.txt"
        dets = get_detections(det_path, frame_number, width, height)
        pred_dets = [
            track.predicted_loc
            for _, track in tracks.items()
            if track.status != Status.Stoped
        ]
        ids1, ids2 = match_two_detection_sets(pred_dets, dets)

        flows = [Point(x=0.0, y=0.0)]
        for id1, id2 in zip(ids1, ids2):
            current_track_id = pred_dets[id1].id
            track = tracks[current_track_id]
            # kill tracks that are not tracked for a while
            if frame_number - track.frameids[-1] > stopped_track_length:
                track.status = Status.Stoped
            else:
                dist = np.linalg.norm(
                    [pred_dets[id1].x - dets[id2].x, pred_dets[id1].y - dets[id2].y]
                )
                if dist < accepted_flow_length:
                    track.coords.append(dets[id2])
                    track.frameids.append(frame_number)
                    flow = Point(
                        x=track.coords[-1].x - track.coords[-2].x,
                        y=track.coords[-1].y - track.coords[-2].y,
                    )
                    flow_length = np.linalg.norm([flow.x, flow.y])
                    if flow_length < accepted_flow_length:
                        flows.append(flow)
                    pred = Point(
                        x=flow.x + track.coords[-1].x, y=flow.y + track.coords[-1].y
                    )
                    track.predicted_loc.x = pred.x
                    track.predicted_loc.y = pred.y
                    track.status = Status.Tracked
                else:
                    track.predicted_loc.x = common_flow.x + track.predicted_loc.x
                    track.predicted_loc.y = common_flow.y + track.predicted_loc.y
                    if track.status != Status.NewTrack:
                        track.status = Status.Untracked
        common_flow = _get_common_flow(flows)

        # unmatched tracks: frame1
        diff_ids = set(range(len(pred_dets))).difference(set(ids1))
        for id in diff_ids:
            current_track_id = pred_dets[id].id
            track = tracks[current_track_id]
            track.predicted_loc.x += common_flow.x
            track.predicted_loc.y += common_flow.y
            if track.status != Status.NewTrack:
                track.status = Status.Untracked

        # unmatched tracks: frame2
        diff_ids = set(range(len(dets))).difference(set(ids2))
        for id in diff_ids:
            coords = [dets2[id]]
            frameids = [frame_number]

            tracks[track_id] = _make_a_new_track(
                coords, frameids, common_flow, track_id, Status.NewTrack
            )
            track_id += 1
    return tracks
