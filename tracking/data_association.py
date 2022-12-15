import enum
import shutil
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


@dataclass
class DispWithProb:
    val: int = -1
    prob: float = 0.0
    count: int = 0
    current_frame_number: int = 0
    frame_number: int = 0


@dataclass
class DetectionWithDisp:
    x: int
    y: int
    w: int
    h: int
    det_id: int
    disp_candidates: list[int]
    frame_number: int = -1
    score: np.float16 = -1
    camera_id: int = 0


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    det_id: int
    frame_number: int = -1
    score: np.float16 = -1
    camera_id: int = 0
    # track_id: int = -1


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
    det_file: Path, width: int, height: int, camera_id: int = 1
) -> list[Detection]:
    frame_number = int(det_file.stem.split("_")[-1])
    detections = np.loadtxt(det_file)
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


def _tl_br_from_cen_wh(center_x, center_y, bbox_w, bbox_h) -> tuple:
    return (
        int(round(center_x - bbox_w / 2)),
        int(round(center_y - bbox_h / 2)),
        int(round(center_x + bbox_w / 2)),
        int(round(center_y + bbox_h / 2)),
    )


def get_detections_array(det_file: Path, width: int, height: int) -> list[np.ndarray]:
    frame_number = int(det_file.stem.split("_")[-1])
    detections = np.loadtxt(det_file)
    dets_array = []
    for det_id, det in enumerate(detections):
        center_x = int(round(det[1] * width))
        center_y = int(round(det[2] * height))
        bbox_width = int(round(det[3] * width))
        bbox_height = int(round(det[4] * height))
        x_tl, y_tl, x_br, y_br = _tl_br_from_cen_wh(
            center_x, center_y, bbox_width, bbox_height
        )

        item = [
            det_id,
            frame_number - 1,
            0,
            x_tl,
            y_tl,
            x_br,
            y_br,
            center_x,
            center_y,
            bbox_width,
            bbox_height,
        ]
        dets_array.append(item)
    return np.array(dets_array).astype(np.int64)


def make_array_from_dets(dets: list[Detection]):
    # array format: det_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    dets_array = []
    for det in dets:
        x_tl, y_tl, x_br, y_br = _tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
        item = [
            det.det_id,
            det.frame_number - 1,
            0,
            x_tl,  # top left
            y_tl,
            x_br,  # bottom right
            y_br,
            int(round(det.x)),  # center
            int(round(det.y)),
            det.w,
            det.h,
        ]
        dets_array.append(item)
    return np.array(dets_array).astype(np.int64)


def make_array_from_tracks(tracks) -> np.ndarray:
    # # array format: track_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    tracks_array = []
    for track_id, track in tracks.items():
        for det in track.coords:
            x_tl, y_tl, x_br, y_br = _tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
            item = [
                track_id,
                det.frame_number - 1,  # my frame_number starts from 1
                det.det_id,
                x_tl,  # top left
                y_tl,
                x_br,  # bottom right
                y_br,
                int(round(det.x)),  # center
                int(round(det.y)),
                det.w,
                det.h,
            ]
            tracks_array.append(item)
    return np.array(tracks_array).astype(np.int64)


def make_dets_from_array(dets_array: np.ndarray) -> list[Detection]:
    # array format: det_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    dets = []
    for det in dets_array:
        item = Detection(
            x=det[7],
            y=det[8],
            w=det[9],
            h=det[10],
            det_id=det[0],
            frame_number=det[1] + 1,
        )
        dets.append(item)
    return dets


def _get_dets_from_indices_of_array(idxs, annos: np.ndarray):
    dets_anno = []
    for idx in idxs:
        anno = annos[idx]
        det = Detection(
            x=anno[7],
            y=anno[8],
            w=anno[9],
            h=anno[10],
            det_id=anno[0],
            frame_number=anno[1] + 1,  # my frame_number starts from 1
        )
        dets_anno.append(det)
    return dets_anno


def _get_track_coords_from_array(annos: np.ndarray, track_id: int):
    idxs = np.where(annos[:, 0] == track_id)[0]
    assert len(idxs) != 0, "this track doesn't exist"
    return _get_dets_from_indices_of_array(idxs, annos)


def make_tracks_from_array(annos: np.ndarray):
    tracks_anno = {}
    track_ids = np.unique(annos[:, 0])
    for track_id in track_ids:
        dummy_prediction = Prediction(
            -1, -1, -1, -1, track_id=track_id, det_id=-1, frame_number=-1
        )
        coords = _get_track_coords_from_array(annos, track_id)
        color = tuple(np.random.rand(3).astype(np.float16))
        tracks_anno[track_id] = Track(
            coords, dummy_prediction, color=color, status=Status.Tracked
        )
    return tracks_anno


def _find_dets_around_a_det(det: np.ndarray, dets: np.ndarray, thres=20):
    candidates = dets[
        ((abs(dets[:, 3] - det[3]) < thres) & (abs(dets[:, 4] - det[4]) < thres))
        | ((abs(dets[:, 5] - det[5]) < thres) & (abs(dets[:, 6] - det[6]) < thres))
    ]
    return candidates


def clean_detections(dets: np.ndarray, ratio_thres=2.5, thres=20, inters_thres=0.85):
    remove_idxs = []
    for idx, det in enumerate(dets):
        # remove based on shape of the bbox
        if det[9] / det[10] > ratio_thres:
            remove_idxs.append(idx)

        # remove if other detection if one detection is within the other detection
        candidates = _find_dets_around_a_det(det, dets, thres)
        idx_det = np.where(candidates[:, 0] == det[0])[0][0]
        candidates = np.delete(candidates, idx_det, axis=0)
        for item in candidates:
            if is_bbox_in_bbox(det[3:7], item[3:7], inters_thres):
                rem_idx = np.where(dets[:, 0] == item[0])[0][0]
                remove_idxs.append(rem_idx)
    dets = np.delete(dets, remove_idxs, axis=0)
    return dets


def match_detection(det1, dets2, thres=20, min_iou=0.1):
    candidates = _find_dets_around_a_det(det1, dets2, thres)
    if len(candidates) == 0:
        return None
    ious = []
    for det in candidates:
        ious.append(get_iou(det1[3:7], det[3:7]))
    ious = np.array(ious)

    # remove the occluded detections
    if len(ious[ious > 0]) > 1:
        return None

    iou_max = max(ious)
    if iou_max < min_iou:
        return None
    idx = np.where(ious == iou_max)[0][0]
    return candidates[idx]


def _get_indices(dets, det_ids):
    idxs = []
    for det_id in det_ids:
        idxs.append(np.where(dets[:, 0] == det_id)[0][0])
    return np.array(idxs)


def match_detections(dets1: np.ndarray, dets2: np.ndarray):
    matched_ids = []
    for det1 in dets1:
        det2 = match_detection(det1, dets2)
        if det2 is not None:
            matched_ids.append([det1[0], det2[0]])
    matched_ids = np.array(matched_ids).astype(np.int64)
    if len(matched_ids.shape) == 1:
        return None, None, None
    idxs1 = _get_indices(dets1, matched_ids[:, 0])
    idxs2 = _get_indices(dets2, matched_ids[:, 1])
    return idxs1, idxs2, matched_ids


def match_two_detection_sets_with_disps(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - get_iou(_get_tl_and_br(det1), _get_tl_and_br(det2))
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


def _get_tl_and_br(det: Detection) -> tuple:
    return det.x - det.w / 2, det.y - det.h / 2, det.x + det.w / 2, det.y + det.h / 2


def match_two_detection_sets(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):

            iou_loss = 1 - get_iou(_get_tl_and_br(det1), _get_tl_and_br(det2))
            loc_loss = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
            dist[i, j] = iou_loss + loc_loss
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def _connect_idxs_to_detection_ids(dets):
    idxs_to_det_ids = {i: det.det_id for i, det in enumerate(dets)}
    det_ids_to_idxs = {det.det_id: id for i, det in enumerate(dets)}
    return idxs_to_det_ids, det_ids_to_idxs


def _make_a_new_track(coords: list[Detection], new_track_id) -> Track:
    color = tuple(np.random.rand(3).astype(np.float16))
    predicted_loc = Prediction(
        x=coords[-1].x,
        y=coords[-1].y,
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
        status=Status.NewTrack,
    )
    return track


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


def _initialize_track_frame1(dets):
    new_track_id = 0
    tracks = {}
    ids = range(len(dets))
    for id in ids:
        coords = [dets[id]]

        tracks[new_track_id] = _make_a_new_track(coords, new_track_id)
        new_track_id += 1
    return tracks, new_track_id


def initialize_tracks(det_folder: Path, filename_fixpart: str, width: int, height: int):
    frame_number1 = 1
    det_path = det_folder / f"{filename_fixpart}_{frame_number1}.txt"
    dets = get_detections(det_path, width, height)
    tracks, new_track_id = _initialize_track_frame1(dets)
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
    frame_number = 1
    det_path_cam1 = det_folder_cam1 / f"{filename_fixpart_cam1}_{frame_number}.txt"
    det_path_cam2 = det_folder_cam2 / f"{filename_fixpart_cam2}_{frame_number}.txt"
    dets = get_detections_with_disp(
        det_path_cam1,
        det_path_cam2,
        width,
        height,
        camera_id,
    )
    tracks, new_track_id = _initialize_track_frame1(dets)

    # assign a unique disparity: heuristics
    tracks = _assign_unique_disp(tracks, frame_number)

    # assign disparities to predictions
    for _, track in tracks.items():
        track.predicted_loc.disp = track.disp

    return tracks, new_track_id


def _get_predicted_flow(track, current_frame_number):
    flow = Point(0, 0)
    if len(track.coords) > 1:
        num_missing_frames = current_frame_number - track.coords[-1].frame_number
        flow = Point(
            x=(track.coords[-1].x - track.coords[-2].x) * num_missing_frames // 2,
            y=(track.coords[-1].y - track.coords[-2].y) * num_missing_frames // 2,
        )
    return flow


def _track_predicted_unmatched(pred_dets, pred_ids, tracks):
    diff_ids = set(range(len(pred_dets))).difference(set(pred_ids))
    for id in diff_ids:
        current_track_id = pred_dets[id].track_id
        track = tracks[current_track_id]
        track.predicted_loc = _update_pred_loc(track.predicted_loc, flow=Point(0, 0))
        track.status = Status.Untracked
    return tracks


def _track_current_unmatched(dets, ids, frame_number, tracks, new_track_id):
    diff_ids = set(range(len(dets))).difference(set(ids))
    for id in diff_ids:
        tracks[new_track_id] = _make_a_new_track([dets[id]], new_track_id)
        new_track_id += 1
    return tracks, new_track_id


def _track_matches(
    pred_ids, ids, pred_dets, dets, tracks, current_frame_number, new_track_id
):
    for id1, id2 in zip(pred_ids, ids):
        current_track_id = pred_dets[id1].track_id
        track = tracks[current_track_id]
        # kill tracks that are not tracked for a while
        if current_frame_number - track.coords[-1].frame_number > stopped_track_length:
            track.status = Status.Stoped
        else:
            dist = np.linalg.norm(
                [pred_dets[id1].x - dets[id2].x, pred_dets[id1].y - dets[id2].y]
            )
            if dist < accepted_flow_length:
                track.coords.append(dets[id2])
                predicted_loc = _make_pred_loc_from_det(
                    track.coords[-1], current_track_id
                )

                flow = _get_predicted_flow(track, current_frame_number)
                track.predicted_loc = _update_pred_loc(predicted_loc, flow)
                track.status = Status.Tracked
            else:
                track.predicted_loc = _update_pred_loc(
                    track.predicted_loc, flow=Point(0, 0)
                )
                track.status = Status.Untracked
                # # bug resolve: but generates many tracklets
                # tracks[new_track_id] = _make_a_new_track(
                #     [dets[id2]], new_track_id
                # )
                # new_track_id += 1
    return tracks, new_track_id


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
    tracks, new_track_id = initialize_tracks_with_disps(
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
    for frame_number in range(2, total_no_frames + 1):
        det_path_cam1 = det_folder_cam1 / f"{filename_fixpart_cam1}_{frame_number}.txt"
        det_path_cam2 = det_folder_cam2 / f"{filename_fixpart_cam2}_{frame_number}.txt"
        dets = get_detections_with_disp(
            det_path_cam1,
            det_path_cam2,
            width,
            height,
            camera_id,
        )

        pred_dets = [
            track.predicted_loc
            for _, track in tracks.items()
            if track.status != Status.Stoped
        ]
        pred_ids, ids = match_two_detection_sets_with_disps(pred_dets, dets)

        # track maches
        tracks, new_track_id = _track_matches(
            pred_ids,
            ids,
            pred_dets,
            dets,
            tracks,
            frame_number,
            new_track_id,
        )

        # unmatched tracks: predicted
        tracks = _track_predicted_unmatched(pred_dets, pred_ids, tracks)

        # unmatched tracks: current
        tracks, new_track_id = _track_current_unmatched(
            dets, ids, frame_number, tracks, new_track_id
        )

        # assign a unique disparity: heuristics
        tracks = _assign_unique_disp(tracks, frame_number)

        # assign disparities to predictions
        for _, track in tracks.items():
            track.predicted_loc.disp = track.disp

    return tracks


def compute_tracks(
    det_folder: Path,
    filename_fixpart: str,
    camera_id: int,
    width: int,
    height: int,
    total_no_frames: int,
):
    tracks, new_track_id = initialize_tracks(
        det_folder, filename_fixpart, width, height
    )

    # start track
    # ===========
    for frame_number in range(2, total_no_frames + 1):
        det_path = det_folder / f"{filename_fixpart}_{frame_number}.txt"
        dets = get_detections(det_path, width, height, camera_id)
        pred_dets = [
            track.predicted_loc
            for _, track in tracks.items()
            if track.status != Status.Stoped
        ]
        pred_ids, ids = match_two_detection_sets(pred_dets, dets)
        # print(frame_number)
        # print(pred_ids.shape, ids.shape)
        # print(np.vstack((pred_ids, ids)).T)
        ## TODO: bug somewhere
        # pred_dets_array = make_array_from_dets(pred_dets)
        # dets_cleaned = make_array_from_dets(
        #     dets
        # )  # clean_detections(make_array_from_dets(dets))
        # dets = make_dets_from_array(dets_cleaned)
        # pred_ids, ids, _ = match_detections(pred_dets_array, dets_cleaned)

        if ids is None:
            return tracks

        # track maches
        tracks, new_track_id = _track_matches(
            pred_ids,
            ids,
            pred_dets,
            dets,
            tracks,
            frame_number,
            new_track_id,
        )

        # unmatched tracks: predicted
        tracks = _track_predicted_unmatched(pred_dets, pred_ids, tracks)

        # unmatched tracks: current
        tracks, new_track_id = _track_current_unmatched(
            dets, ids, frame_number, tracks, new_track_id
        )

        if frame_number in [46, 47]:
            print(tracks[35].coords[-1])
            print(tracks[35].predicted_loc)
            print("=====")
    return tracks


def _rm_det_chang_track_id(tracks: np.ndarray, frame_number: int, track_id: int):
    latest_track_id = np.unique(np.sort(tracks[:, 0]))[-1]
    idx1 = np.where((tracks[:, 1] == frame_number) & (tracks[:, 0] == track_id))[0][0]
    idxs = np.where((tracks[:, 1] > frame_number) & (tracks[:, 0] == track_id))[0]
    if len(idxs) != 0:
        tracks[idxs, 0] = latest_track_id + 1
    tracks = np.delete(tracks, idx1, axis=0)
    return tracks


def remove_detect_change_track_id_per_frame(tracks: np.ndarray, frame_number: int):
    frame_tracks = tracks[tracks[:, 1] == frame_number].copy()
    track_ids_remove = []
    for i in range(len(frame_tracks)):
        for j in range(i + 1, len(frame_tracks)):
            item1 = frame_tracks[i]
            item2 = frame_tracks[j]
            iou = get_iou(item1[3:7], item2[3:7])
            if iou > 0:
                track_ids_remove.append(item1[0])
                track_ids_remove.append(item2[0])

    track_ids_remove = list(set(track_ids_remove))
    for track_id in track_ids_remove:
        tracks = _rm_det_chang_track_id(tracks, frame_number, track_id)
    return tracks


def remove_detects_change_track_ids(tracks: np.ndarray):
    frame_numbers = np.unique(np.sort(tracks[:, 1]))
    for frame_number in frame_numbers:
        tracks = remove_detect_change_track_id_per_frame(tracks, frame_number)
    return tracks


def remove_short_tracks(tracks: np.ndarray, min_track_length: int = 50):
    track_ids = np.unique(np.sort(tracks[:, 0]))
    for track_id in track_ids:
        idxs = np.where(tracks[:, 0] == track_id)[0]
        if len(idxs) < min_track_length:
            tracks = np.delete(tracks, idxs, axis=0)
    return tracks


def arrange_track_ids(tracks: np.ndarray):
    new_tracks = tracks.copy()
    track_ids = np.unique(np.sort(tracks[:, 0]))
    old_to_new_ids = {
        track_id: new_track_id for new_track_id, track_id in enumerate(track_ids)
    }
    for track_id in track_ids:
        new_tracks[tracks[:, 0] == track_id, 0] = old_to_new_ids[track_id]
    return new_tracks


def save_tracks_to_mot_format(
    save_file: Path, tracks: np.ndarray | dict[Track], make_zip: bool = True
):
    """MOT format is 1-based, including bbox. https://arxiv.org/abs/2003.09003
    mot format: frame_id, track_id, xtl, ytl, w, h, score, class, visibility
    array format: track_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    """
    track_folder = save_file.parent / "gt"
    track_folder.mkdir(parents=True, exist_ok=True)
    with open(track_folder / "labels.txt", "w") as wf:
        wf.write("fish")

    track_file = track_folder / "gt.txt"
    if isinstance(tracks, dict):
        with open(track_file, "w") as file:
            for track_id, track in tracks.items():
                for det in track.coords:
                    top_left_x = det.x - det.w / 2
                    top_left_y = det.y - det.h / 2
                    file.write(
                        f"{det.frame_number},{track_id+1},{top_left_x+1},{top_left_y+1},{det.w},{det.h},1,1,1.0\n"
                    )
    if isinstance(tracks, np.ndarray):
        with open(track_file, "w") as file:
            for item in tracks:
                file.write(
                    f"{int(item[1])+1},{int(item[0])+1},{item[3]+1},{item[4]+1},{int(item[9])},{int(item[10])},1,1,1.0\n"
                )
    if make_zip:
        shutil.make_archive(save_file, "zip", save_file.parent, "gt")
        shutil.rmtree(track_folder)


def read_tracks_from_mot_format(track_file: Path) -> np.ndarray:
    """
    mot format: frame_id, track_id, xtl, ytl, w, h, score, class, visibility
    array format: track_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    """
    tracks = []
    with open(track_file, "r") as file:
        for row in file:
            items = row.split("\n")[0].split(",")
            top_left_x, top_left_y, width, height = (
                float(items[2]) - 1,
                float(items[3]) - 1,
                int(items[4]),
                int(items[5]),
            )
            center_x = top_left_x + width / 2
            center_y = top_left_y + height / 2
            bottom_right_x = top_left_x + width
            bottom_right_y = top_left_y + height
            track = [
                int(items[1]) - 1,
                int(items[0]) - 1,
                0,
                top_left_x,
                top_left_y,
                bottom_right_x,
                bottom_right_y,
                center_x,
                center_y,
                width,
                height,
            ]
            tracks.append(track)
    return np.array(tracks).astype(np.int64)


def read_tracks_cvat_txt_format(track_file: Path) -> np.ndarray:
    """
    array format: track_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    """
    tracks = np.round(np.loadtxt(track_file, skiprows=1, delimiter=",")).astype(
        np.int64
    )
    centers_x = np.int64(np.round((tracks[:, 5] + tracks[:, 3]) / 2)).reshape(-1, 1)
    centers_y = np.int64(np.round((tracks[:, 6] + tracks[:, 4]) / 2)).reshape(-1, 1)
    width = np.int64(np.round(tracks[:, 5] - tracks[:, 3])).reshape(-1, 1)
    height = np.int64(np.round(tracks[:, 6] - tracks[:, 4])).reshape(-1, 1)
    return np.concatenate((tracks, centers_x, centers_y, width, height), axis=1)


def save_tracks_cvat_txt_format(track_file: Path, tracks: np.ndarray):
    np.savetxt(
        track_file,
        tracks[:, :7],
        header="IDs,frames,outside,xtl,ytl,xbr,ybr",
        delimiter=",",
        fmt="%d",
    )


def save_tracks(track_file: Path, tracks: np.ndarray | dict[Track]):
    if isinstance(tracks, dict):
        with open(track_file, "w") as file:
            for track_id, track in tracks.items():
                for det in track.coords:
                    file.write(
                        f"{track_id},{det.frame_number},{det.det_id},{det.x},{det.y},{det.w},{det.h},{det.score:.2f},{track.status.value}\n"
                    )

    if isinstance(tracks, np.ndarray):
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
    # det1,2: (x_topleft,y_topleft, x_bottomright, y_bottomright)
    # copied and modified from
    # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    # determine the coordinates of the intersection rectangle
    x_left = max(det1[0], det2[0])
    y_top = max(det1[1], det2[1])
    x_right = min(det1[2], det2[2])
    y_bottom = min(det1[3], det2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
    area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])

    iou = intersection_area / float(area1 + area2 - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_bbox_in_bbox(det1, det2, inters_thres=0.85) -> float:
    # det1,2: (x_topleft,y_topleft, x_bottomright, y_bottomright)

    # determine the coordinates of the intersection rectangle
    x_left = max(det1[0], det2[0])
    y_top = max(det1[1], det2[1])
    x_right = min(det1[2], det2[2])
    y_bottom = min(det1[3], det2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
    area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])

    inters_ratio1 = intersection_area / float(area1)
    inters_ratio2 = intersection_area / float(area2)
    if (inters_ratio1 >= inters_ratio2) & (inters_ratio1 > inters_thres):
        return True
    else:
        return False


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


def find_track_id_by_coord_and_frame_number(tracks, x, y, frame_number, tolerance=3):
    for track_id, track in tracks.items():
        det = find_detection_in_track_by_frame_number(track, frame_number)
        if det:
            if (abs(det.x - x) < tolerance) & (abs(det.y - y) < tolerance):
                return track_id
