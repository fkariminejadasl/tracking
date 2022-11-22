import enum
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

np.random.seed(1000)

accepted_flow_length = 10
stopped_track_length = 50
accepted_rect_error = 3
smallest_disparity = 250
largest_disparity = 650


def get_video_parameters(vc: cv2.VideoCapture):
    if vc.isOpened():
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vc.get(cv2.CAP_PROP_FPS)
        return height, width, total_no_frames, fps
    else:
        return


class Status(enum.Enum):
    Tracked: bool = enum.auto()
    Untracked: bool = enum.auto()
    Stoped: bool = enum.auto()
    NewTrack: bool = enum.auto()


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, scale: float):
        return Point(x=self.x * scale, y=self.y * scale)


@dataclass(kw_only=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    det_id: int
    frame_number: int = -1
    score: np.float16 = -1
    camera_id: int = 0


def _compute_disp_candidates(det1, dets2) -> list[int]:
    disp_candidates = []
    for det2 in dets2:
        disp = abs(det1.x - det2.x)
        rectification_error = abs(det1.y - det2.y)
        if (
            rectification_error < accepted_rect_error
            and disp < largest_disparity
            and disp > smallest_disparity
        ):
            disp_candidates.append(disp)
    return disp_candidates


@dataclass
class DispWithProb:
    val: int = -1
    prob: float = 0.0
    count: int = 0
    current_frame_number: int = 0
    frame_number: int = 0


@dataclass(kw_only=True)
class DetectionWithDisp(Detection):
    disp_candidates: list[int]


def get_detections_with_disp(
    det_path_cam1,
    det_path_cam2,
    width: int,
    height: int,
    camera_id: int = 1,
) -> list[DetectionWithDisp]:
    camera_id2 = 2
    if camera_id == 2:
        camera_id2 = 1
    frame_number = int(det_path_cam1.stem.split("_")[-1])
    assert frame_number == int(det_path_cam2.stem.split("_")[-1]), "not a stereo pair"
    dets_cam1 = get_detections(det_path_cam1, width, height, camera_id=camera_id)
    dets_cam2 = get_detections(det_path_cam2, width, height, camera_id=camera_id2)
    detections = []
    for det in dets_cam1:
        disp_candidates = _compute_disp_candidates(det, dets_cam2)
        detection = DetectionWithDisp(
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            frame_number=frame_number,
            det_id=det.det_id,
            score=det.score,
            camera_id=det.camera_id,
            disp_candidates=disp_candidates,
        )
        detections.append(detection)

    return detections


def get_detections(
    det_path, width: int, height: int, camera_id: int = 1
) -> list[Detection]:
    frame_number = int(det_path.stem.split("_")[-1])
    detections = np.loadtxt(det_path)
    return [
        Detection(
            x=int(round(det[1] * width)),
            y=int(round(det[2] * height)),
            w=int(round(det[3] * width)),
            h=int(round(det[4] * height)),
            frame_number=frame_number,
            det_id=det_id,
            score=np.float16(f"{det[5]:.2f}"),
            camera_id=camera_id,
        )
        for det_id, det in enumerate(detections)
    ]


@dataclass
class Prediction:
    x: int
    y: int
    w: int
    h: int
    track_id: int
    det_id: int
    frame_number: int
    disp: DispWithProb = DispWithProb()


@dataclass
class Track:
    coords: list[Detection]
    predicted_loc: Prediction
    color: tuple
    status: Status
    disp: DispWithProb = DispWithProb()


# def match_two_detection_sets_with_disps(dets1, dets2):
#     ignored_inds = []
#     if isinstance(dets1[0], DetectionWithDisp):
#         for det in dets2:
#             if len(det.disp_candidates)==0:
#                 ignored_inds.append(det.det_id)
#     dist = np.zeros((len(dets1), len(dets2)-len(ignored_inds)), dtype=np.float32)
#     for det1 in dets1:
#         for det2 in dets2:
#             iou_loss = 1 - get_iou(det1, det2)
#             loc_loss = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
#             disp_loss = 0
#             # if det1.disp.val != -1 and len(det2.disp_candidates) > 0:
#             #     disp_loss = min(
#             #         [abs(disp2 - det1.disp.val) for disp2 in det2.disp_candidates]
#             #     )
#             # else:
#             #     disp_loss = 0
#             dist[i, j] = iou_loss + disp_loss
#     row_ind, col_ind = linear_sum_assignment(dist)
#     return row_ind, col_ind
def match_two_detection_sets_with_disps(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - get_iou(det1, det2)
            loc_loss = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
            disp_loss = 0
            # if det1.disp.val != -1 and len(det2.disp_candidates) > 0:
            #     disp_loss = min(
            #         [abs(disp2 - det1.disp.val) for disp2 in det2.disp_candidates]
            #     )
            # else:
            #     disp_loss = 0
            dist[i, j] = iou_loss + disp_loss
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def match_two_detection_sets(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - get_iou(det1, det2)
            loc_loss = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
            dist[i, j] = iou_loss + loc_loss
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def _connect_idxs_to_detection_ids(dets):
    idxs_to_det_ids = {i: det.det_id for i, det in enumerate(dets)}
    det_ids_to_idxs = {det.det_id: id for i, det in enumerate(dets)}
    return idxs_to_det_ids, det_ids_to_idxs


def _make_a_new_track(
    coords: list[Detection], flow: Point, new_track_id, status
) -> Track:
    color = tuple(np.random.rand(3).astype(np.float16))
    pred = Point(x=flow.x + coords[-1].x, y=flow.y + coords[-1].y)
    predicted_loc = Prediction(
        x=flow.x + coords[-1].x,
        y=flow.y + coords[-1].y,
        w=coords[-1].w,
        h=coords[-1].h,
        track_id=new_track_id,
        det_id=coords[-1].det_id,
        frame_number=coords[-1].frame_number,
    )
    track = Track(
        coords=coords,
        predicted_loc=predicted_loc,
        color=color,
        status=status,
    )
    return track


def _get_common_flow(flows):
    common_flow = Point(
        x=np.average(np.array([flow.x for flow in flows])),
        y=np.average(np.array([flow.y for flow in flows])),
    )
    return common_flow


def _make_pred_loc_from_det(det, track_id: int = -1):
    return Prediction(
        det.x, det.y, det.w, det.h, track_id, det.det_id, det.frame_number
    )


def _update_pred_loc(predicted_loc, flow):
    updated_predicted_loc = Prediction(
        x=predicted_loc.x + flow.x,
        y=predicted_loc.y + flow.y,
        w=predicted_loc.w,
        h=predicted_loc.h,
        track_id=predicted_loc.track_id,
        det_id=predicted_loc.det_id,
        frame_number=predicted_loc.frame_number,
    )
    return updated_predicted_loc


def _initialize_matches(ids1, ids2, dets1, dets2, frame_number1, frame_number2):
    tracks = {}
    new_track_id = 0
    flows = [Point(x=0.0, y=0.0)]
    for id1, id2 in zip(ids1, ids2):
        dist = np.linalg.norm(
            [dets1[id1].x - dets2[id2].x, dets1[id1].y - dets2[id2].y]
        )
        if dist < accepted_flow_length:
            coords = [dets1[id1], dets2[id2]]
            flow = Point(x=coords[-1].x - coords[-2].x, y=coords[-1].y - coords[-2].y)
            flow_length = np.linalg.norm([flow.x, flow.y])
            if flow_length < accepted_flow_length:
                flows.append(flow)
            tracks[new_track_id] = _make_a_new_track(
                coords, flow, new_track_id, Status.Tracked
            )
            new_track_id += 1
        else:
            common_flow = _get_common_flow(flows)
            tracks[new_track_id] = _make_a_new_track(
                [dets1[id1]],
                common_flow * 2,
                new_track_id,
                Status.NewTrack,
            )
            new_track_id += 1
            tracks[new_track_id] = _make_a_new_track(
                [dets2[id2]], common_flow, new_track_id, Status.NewTrack
            )
            new_track_id += 1

    common_flow = _get_common_flow(flows)
    return tracks, new_track_id, common_flow


def _initialize_unmatched_frame1(
    dets1, ids1, frame_number1, common_flow, tracks, new_track_id
):
    diff_ids = set(range(len(dets1))).difference(set(ids1))
    for id in diff_ids:
        coords = [dets1[id]]

        tracks[new_track_id] = _make_a_new_track(
            coords, common_flow * 2, new_track_id, Status.NewTrack
        )
        new_track_id += 1
    return tracks, new_track_id


def _initialize_unmatched_frame2(
    dets2, ids2, frame_number2, common_flow, tracks, new_track_id
):
    diff_ids = set(range(len(dets2))).difference(set(ids2))
    for id in diff_ids:
        coords = [dets2[id]]

        tracks[new_track_id] = _make_a_new_track(
            coords, common_flow, new_track_id, Status.NewTrack
        )
        new_track_id += 1
    return tracks, new_track_id


def _assign_unique_disp(tracks, frame_number):
    for track_id, track in tracks.items():
        coord = track.coords[-1]
        if (len(coord.disp_candidates) == 1) and (frame_number == coord.frame_number):
            disp = DispWithProb(
                coord.disp_candidates[0],
                1.0,
                track.disp.count + 1,
                frame_number,
                coord.frame_number,
            )
            track.disp = disp
        if frame_number != coord.frame_number:
            track.disp.current_frame_number = frame_number
    return tracks


def initialize_tracks_with_disps(
    det_folder_cam1: Path,
    filename_fixpart_cam1: str,
    det_folder_cam2: Path,
    filename_fixpart_cam2: str,
    camera_id: int,
    width: int,
    height: int,
):
    frame_number1 = 1
    frame_number2 = 2
    det_path1_cam1 = det_folder_cam1 / f"{filename_fixpart_cam1}_{frame_number1}.txt"
    det_path2_cam1 = det_folder_cam1 / f"{filename_fixpart_cam1}_{frame_number2}.txt"
    det_path1_cam2 = det_folder_cam2 / f"{filename_fixpart_cam2}_{frame_number1}.txt"
    det_path2_cam2 = det_folder_cam2 / f"{filename_fixpart_cam2}_{frame_number2}.txt"
    dets1 = get_detections_with_disp(
        det_path1_cam1,
        det_path1_cam2,
        width,
        height,
        camera_id,
    )
    dets2 = get_detections_with_disp(
        det_path2_cam1,
        det_path2_cam2,
        width,
        height,
        camera_id,
    )
    ids1, ids2 = match_two_detection_sets(dets1, dets2)
    # matched tracks
    tracks, new_track_id, common_flow = _initialize_matches(
        ids1, ids2, dets1, dets2, frame_number1, frame_number2
    )
    # unmatched tracks: frame1
    tracks, new_track_id = _initialize_unmatched_frame1(
        dets1, ids1, frame_number1, common_flow, tracks, new_track_id
    )
    # unmatched tracks: frame2
    tracks, new_track_id = _initialize_unmatched_frame2(
        dets2, ids2, frame_number2, common_flow, tracks, new_track_id
    )

    # assign a unique disparity: heuristics
    tracks = _assign_unique_disp(tracks, frame_number2)

    # assign disparities to predictions
    for _, track in tracks.items():
        track.predicted_loc.disp = track.disp

    return tracks, common_flow, new_track_id


def initialize_tracks(det_folder: Path, filename_fixpart: str, width: int, height: int):
    frame_number1 = 1
    frame_number2 = 2
    det_path1 = det_folder / f"{filename_fixpart}_{frame_number1}.txt"
    det_path2 = det_folder / f"{filename_fixpart}_{frame_number2}.txt"
    dets1 = get_detections(det_path1, width, height)
    dets2 = get_detections(det_path2, width, height)
    ids1, ids2 = match_two_detection_sets(dets1, dets2)
    # matched tracks
    tracks, new_track_id, common_flow = _initialize_matches(
        ids1, ids2, dets1, dets2, frame_number1, frame_number2
    )
    # unmatched tracks: frame1
    tracks, new_track_id = _initialize_unmatched_frame1(
        dets1, ids1, frame_number1, common_flow, tracks, new_track_id
    )
    # unmatched tracks: frame2
    tracks, new_track_id = _initialize_unmatched_frame2(
        dets2, ids2, frame_number2, common_flow, tracks, new_track_id
    )
    return tracks, common_flow, new_track_id


def _track_predicted_unmatched(pred_dets, pred_ids, tracks, common_flow):
    diff_ids = set(range(len(pred_dets))).difference(set(pred_ids))
    for id in diff_ids:
        current_track_id = pred_dets[id].track_id
        track = tracks[current_track_id]
        track.predicted_loc = _update_pred_loc(track.predicted_loc, common_flow)
        track.status = Status.Untracked
    return tracks


def _track_current_unmatched(
    dets, ids, frame_number, tracks, new_track_id, common_flow
):
    diff_ids = set(range(len(dets))).difference(set(ids))
    for id in diff_ids:
        tracks[new_track_id] = _make_a_new_track(
            [dets[id]], common_flow, new_track_id, Status.NewTrack
        )
        new_track_id += 1
    return tracks, new_track_id


def _track_matches(
    pred_ids, ids, pred_dets, dets, tracks, frame_number, common_flow, new_track_id
):
    flows = [Point(x=0.0, y=0.0)]
    for id1, id2 in zip(pred_ids, ids):
        current_track_id = pred_dets[id1].track_id
        track = tracks[current_track_id]
        # kill tracks that are not tracked for a while
        if frame_number - track.coords[-1].frame_number > stopped_track_length:
            track.status = Status.Stoped
        else:
            dist = np.linalg.norm(
                [pred_dets[id1].x - dets[id2].x, pred_dets[id1].y - dets[id2].y]
            )
            if dist < accepted_flow_length:
                track.coords.append(dets[id2])
                flow = Point(
                    x=track.coords[-1].x - track.coords[-2].x,
                    y=track.coords[-1].y - track.coords[-2].y,
                )
                flow_length = np.linalg.norm([flow.x, flow.y])
                if flow_length < accepted_flow_length:
                    flows.append(flow)
                predicted_loc = _make_pred_loc_from_det(
                    track.coords[-1], current_track_id
                )
                track.predicted_loc = _update_pred_loc(predicted_loc, flow)
                track.status = Status.Tracked
            else:
                track.predicted_loc = _update_pred_loc(track.predicted_loc, common_flow)
                track.status = Status.Untracked
                # # bug resolve: but generates many tracklets
                # tracks[new_track_id] = _make_a_new_track(
                #     [dets[id2]], common_flow, new_track_id, Status.NewTrack
                # )
                # new_track_id += 1
    common_flow = _get_common_flow(flows)
    return tracks, common_flow, new_track_id


def compute_tracks_with_disps(
    det_folder_cam1: Path,
    filename_fixpart_cam1: str,
    det_folder_cam2: Path,
    filename_fixpart_cam2: str,
    camera_id: int,
    width: int,
    height: int,
    total_no_frames: int = 680,
):
    tracks, common_flow, new_track_id = initialize_tracks_with_disps(
        det_folder_cam1,
        filename_fixpart_cam1,
        det_folder_cam2,
        filename_fixpart_cam2,
        camera_id,
        width,
        height,
    )
    # start track
    # ===========
    for frame_number in range(3, total_no_frames + 1):
        det_path_cam1 = det_folder_cam1 / f"{filename_fixpart_cam1}_{frame_number}.txt"
        det_path_cam2 = det_folder_cam2 / f"{filename_fixpart_cam2}_{frame_number}.txt"
        dets = get_detections_with_disp(
            det_path_cam1,
            det_path_cam2,
            width,
            height,
            camera_id,
        )

        # assign disparities to predictions
        for _, track in tracks.items():
            track.predicted_loc.disp = track.disp

        pred_dets = [
            track.predicted_loc
            for _, track in tracks.items()
            if track.status != Status.Stoped
        ]
        pred_ids, ids = match_two_detection_sets_with_disps(pred_dets, dets)

        # track maches
        tracks, common_flow, new_track_id = _track_matches(
            pred_ids,
            ids,
            pred_dets,
            dets,
            tracks,
            frame_number,
            common_flow,
            new_track_id,
        )

        # unmatched tracks: predicted
        tracks = _track_predicted_unmatched(pred_dets, pred_ids, tracks, common_flow)

        # unmatched tracks: current
        tracks, new_track_id = _track_current_unmatched(
            dets, ids, frame_number, tracks, new_track_id, common_flow
        )

        # assign a unique disparity: heuristics
        tracks = _assign_unique_disp(tracks, frame_number)
    return tracks


def compute_tracks(
    det_folder: Path,
    filename_fixpart: str,
    camera_id: int,
    width: int,
    height: int,
    total_no_frames: int,
):
    tracks, common_flow, new_track_id = initialize_tracks(
        det_folder, filename_fixpart, width, height
    )
    # start track
    # ===========
    for frame_number in range(3, total_no_frames + 1):
        det_path = det_folder / f"{filename_fixpart}_{frame_number}.txt"
        dets = get_detections(det_path, width, height, camera_id)
        pred_dets = [
            track.predicted_loc
            for _, track in tracks.items()
            if track.status != Status.Stoped
        ]
        pred_ids, ids = match_two_detection_sets(pred_dets, dets)

        # track maches
        tracks, common_flow, new_track_id = _track_matches(
            pred_ids,
            ids,
            pred_dets,
            dets,
            tracks,
            frame_number,
            common_flow,
            new_track_id,
        )

        # unmatched tracks: predicted
        tracks = _track_predicted_unmatched(pred_dets, pred_ids, tracks, common_flow)

        # unmatched tracks: current
        tracks, new_track_id = _track_current_unmatched(
            dets, ids, frame_number, tracks, new_track_id, common_flow
        )

    return tracks


def save_tracks(track_file, tracks):
    with open(track_file, "w") as file:
        for track_id, track in tracks.items():
            for det in track.coords:
                file.write(
                    f"{track_id},{det.frame_number},{det.det_id},{det.x},{det.y},{det.w},{det.h},{det.score:.2f},{track.status.value}\n"
                )


def write_tracks(track_file, tracks):
    with open(track_file, "w") as file:
        for track in tracks:
            file.write(
                f"{track[0]},{track[1]},{track[2]},{track[3]},{track[4]},{track[5]},{track[6]},{track[7]:.2f},{track[8]}\n"
            )


def read_tracks(track_file):
    tracks = []
    with open(track_file, "r") as f:
        for row in f:
            items = row.split("\n")[0].split(",")
            track_item = [
                int(items[0]),
                int(items[1]),
                int(items[2]),
                int(items[3]),
                int(items[4]),
                int(items[5]),
                int(items[6]),
                float(items[7]),
                int(items[8]),
            ]
            tracks.append(track_item)
    return tracks


def get_iou(det1, det2) -> float:
    # copied from
    # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    # determine the coordinates of the intersection rectangle
    x_left = max(det1.x, det2.x)
    y_top = max(det1.y, det2.y)
    x_right = min(det1.x + det1.w, det2.x + det2.w)
    y_bottom = min(det1.y + det1.h, det2.y + det2.h)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = det1.w * det1.h
    bb2_area = det2.w * det2.h

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def test_get_iou():
    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(2, 1, 3, 2, 1)

    np.testing.assert_almost_equal(get_iou(det1, det2), 0.167, decimal=2)

    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(4, 2, 2, 1, 1)
    np.testing.assert_almost_equal(get_iou(det1, det2), 0.0, decimal=2)


def find_detection_in_track_by_frame_number(track, frame_number):
    for det in track.coords:
        if det.frame_number == frame_number:
            return det


def find_detectios_in_tracks_by_frame_number(tracks, frame_number):
    dets = {}
    for track_id, track in tracks.items():
        det = find_detection_in_track_by_frame_number(track, frame_number)
        if det:
            dets[track_id] = det
    return dets


def get_frame_numbers_of_track(track):
    return [coord.frame_number for coord in track.coords]
