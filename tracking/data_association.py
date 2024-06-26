import csv
import enum
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

np.random.seed(1000)

accepted_flow_length = 10
stopped_track_length = 50
min_track_length = 10

# score_thres = 0.5  # for detection scores
# sp_thres = 20  # for spatial proximity
# inters_thres = 0.85  # for bbox in behind the other bbox
# min_iou = 0
# flow_decay_rate = 0.5
# ratio_thres = 2.5


accepted_rect_error = 3
smallest_disparity = 250
largest_disparity = 650


class Status(enum.Enum):
    Tracked: bool = enum.auto()
    Untracked: bool = enum.auto()
    Stoped: bool = enum.auto()


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, scale: float):
        return Point(x=self.x * scale, y=self.y * scale)


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    det_id: int
    frame_number: int = -1
    score: np.float16 = -1
    track_id: int = -1


def _copy_detection(det: Detection):
    return Detection(
        x=det.x,
        y=det.y,
        w=det.w,
        h=det.h,
        track_id=det.track_id,
        det_id=det.det_id,
        frame_number=det.frame_number,
        score=det.score,
    )


def _copy_detections(dets: List[Detection]):
    new_dets = []
    for det in dets:
        new_det = _copy_detection(det)
        new_dets.append(new_det)
    return new_dets


def _update_det_loc_by_flow(det: Detection, flow: Point):
    return Detection(
        x=det.x + flow.x,
        y=det.y + flow.y,
        w=det.w,
        h=det.h,
        track_id=det.track_id,
        det_id=det.det_id,
        frame_number=det.frame_number,
        score=det.score,
    )


@dataclass
class Track:
    dets: List[Detection]
    color: tuple
    status: Status


def _copy_track(track: Track):
    new_dets = _copy_detections(track.dets)
    return Track(dets=new_dets, color=track.color, status=track.status)


def get_detections(
    det_file: Path,
    width: int,
    height: int,
    frame_number: int,
) -> List[Detection]:
    score = -1

    detections = np.loadtxt(det_file)

    dets = []
    for det_id, det in enumerate(detections):
        if detections.shape[1] == 6:
            score = np.float16(f"{det[5]:.2f}")
        item = Detection(
            x=int(round(det[1] * width)),
            y=int(round(det[2] * height)),
            w=int(round(det[3] * width)),
            h=int(round(det[4] * height)),
            frame_number=frame_number,
            det_id=det_id,
            score=score,
        )
        dets.append(item)
    return dets


def get_detections_array(
    det_file: Path,
    width: int,
    height: int,
    frame_number: int,
) -> List[np.ndarray]:
    detections = np.loadtxt(det_file)
    dets_array = []
    for det_id, det in enumerate(detections):
        center_x = int(round(det[1] * width))
        center_y = int(round(det[2] * height))
        bbox_width = int(round(det[3] * width))
        bbox_height = int(round(det[4] * height))
        det_score = int(round(det[5] * 100))
        x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(
            center_x, center_y, bbox_width, bbox_height
        )

        item = [
            -1,
            frame_number,
            det_id,
            x_tl,
            y_tl,
            x_br,
            y_br,
            center_x,
            center_y,
            bbox_width,
            bbox_height,
            det_score,
        ]
        dets_array.append(item)
    return np.array(dets_array).astype(np.int64)


def make_array_from_dets(dets: List[Detection]):
    # array format: track_id, frame_id, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h
    dets_array = []
    for det in dets:
        x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
        item = [
            det.track_id,
            det.frame_number,
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
        dets_array.append(item)
    return np.array(dets_array).astype(np.int64)


def make_array_from_tracks(tracks) -> np.ndarray:
    # # array format: track_id, frame_id, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h
    tracks_array = []
    for track_id, track in tracks.items():
        for det in track.dets:
            x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
            item = [
                track_id,
                det.frame_number,
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


def make_dets_from_array(dets_array: np.ndarray) -> List[Detection]:
    # array format: track_id, frame_id, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h
    dets = []
    for det in dets_array:
        item = Detection(
            x=det[7],
            y=det[8],
            w=det[9],
            h=det[10],
            det_id=det[2],
            frame_number=det[1],
            track_id=det[0],
        )
        dets.append(item)
    return dets


def tl_br_from_cen_wh(center_x, center_y, bbox_w, bbox_h) -> Tuple:
    return (
        int(round(center_x - bbox_w / 2)),
        int(round(center_y - bbox_h / 2)),
        int(round(center_x + bbox_w / 2)),
        int(round(center_y + bbox_h / 2)),
    )


def cen_wh_from_tl_br(tl_x, tl_y, br_x, br_y) -> Tuple:
    width = int(round(br_x - tl_x))
    height = int(round(br_y - tl_y))
    center_x = int(round(width / 2 + tl_x))
    center_y = int(round(height / 2 + tl_y))
    return center_x, center_y, width, height


def _get_dets_from_indices_of_array(inds, annos: np.ndarray):
    dets_anno = []
    for ind in inds:
        anno = annos[ind]
        det = Detection(
            x=anno[7],
            y=anno[8],
            w=anno[9],
            h=anno[10],
            det_id=anno[2],
            frame_number=anno[1],
            track_id=anno[1],
        )
        dets_anno.append(det)
    return dets_anno


def _get_track_dets_from_array(annos: np.ndarray, track_id: int):
    inds = np.where(annos[:, 0] == track_id)[0]
    assert len(inds) != 0, "this track doesn't exist"
    return _get_dets_from_indices_of_array(inds, annos)


def make_tracks_from_array(annos: np.ndarray):
    tracks_anno = {}
    track_ids = np.unique(annos[:, 0])
    for track_id in track_ids:
        dets = _get_track_dets_from_array(annos, track_id)
        color = tuple(np.random.rand(3).astype(np.float16))
        tracks_anno[track_id] = Track(dets, color=color, status=Status.Tracked)
    return tracks_anno


def clean_detections_by_score(dets: List[Detection], score_thres=0.5):
    cleaned_dets = [det for det in dets if det.score > score_thres]
    return cleaned_dets


def get_cleaned_detections(
    det_path: Path, width, height, frame_number
) -> List[Detection]:
    dets = get_detections(det_path, width, height, frame_number)
    dets = clean_detections_by_score(dets)
    dets = make_dets_from_array(clean_detections(make_array_from_dets(dets)))
    return dets


def clean_detections(dets: np.ndarray, ratio_thres=2.5, sp_thres=20, inters_thres=0.85):
    remove_inds = []
    for ind, det in enumerate(dets):
        # remove based on shape of the bbox
        # if det[9] / det[10] > ratio_thres:
        #     remove_inds.append(ind)

        candidates = _find_dets_around_det(det, dets, sp_thres)
        ind_det = np.where(candidates[:, 2] == det[2])[0][0]
        candidates = np.delete(candidates, ind_det, axis=0)

        # remove if other detection if one detection is within the other detection
        # for item in candidates:
        #     if is_bbox_in_bbox(det[3:7], item[3:7], inters_thres):
        #         rem_ind = np.where(dets[:, 2] == item[2])[0][0]
        #         remove_inds.append(rem_ind)

        # remove overlapping detections
        for item in candidates:
            if get_iou(det[3:7], item[3:7]) > 0:
                rem_ind = np.where(dets[:, 2] == item[2])[0][0]
                remove_inds.append(rem_ind)
                rem_ind = np.where(dets[:, 2] == det[2])[0][0]
                remove_inds.append(rem_ind)
    cleaned_dets = np.delete(dets, remove_inds, axis=0)
    return cleaned_dets


def match_detection(det1, dets2, sp_thres=100, min_iou=0):
    candidates = _find_dets_around_det(det1, dets2, sp_thres)
    if len(candidates) == 0:
        return None
    ious = []
    for det in candidates:
        ious.append(get_iou(det1[3:7], det[3:7]))
    ious = np.array(ious)

    # remove the occluded detections
    # TODO: maybe return inactive
    # if len(ious[ious > 0]) > 1:
    #     return None

    iou_max = max(ious)

    # here no intersection returns none
    if iou_max <= min_iou:
        return None

    ind = np.where(ious == iou_max)[0][0]

    # handle no intersections, evaluate based on location
    if (iou_max == 0) & (len(ious) > 1):
        dists = []
        for det in candidates:
            dists.append(np.linalg.norm(det1[7:9] - det[7:9]))
        dists = np.array(dists)
        ind = np.where(dists == min(dists))[0][0]
    return candidates[ind]


def _find_dets_around_det(det: np.ndarray, dets: np.ndarray, sp_thres=20):
    candidates = dets[
        ((abs(dets[:, 3] - det[3]) < sp_thres) & (abs(dets[:, 4] - det[4]) < sp_thres))
        | (
            (abs(dets[:, 5] - det[5]) < sp_thres)
            & (abs(dets[:, 6] - det[6]) < sp_thres)
        )
    ].copy()
    return candidates


def _get_indices(dets: np.ndarray, ids: np.ndarray):
    inds = []
    for id in ids:
        track_id = id[0]
        det_id = id[1]
        if track_id == -1:
            inds.append(np.where(dets[:, 2] == det_id)[0][0])
        else:
            inds.append(
                np.where((dets[:, 2] == det_id) & (dets[:, 0] == track_id))[0][0]
            )
    return np.array(inds)


def match_detections(dets1: np.ndarray, dets2: np.ndarray):
    matched_ids = []
    for det1 in dets1:
        det2 = match_detection(det1, dets2)
        if det2 is not None:
            matched_ids.append([det1[0], det1[2], det2[0], det2[2]])
    matched_ids = np.array(matched_ids).astype(np.int64)
    if len(matched_ids) == 0:
        return None, None, None
    inds1 = _get_indices(dets1, matched_ids[:, 0:2])
    inds2 = _get_indices(dets2, matched_ids[:, 2:])
    return inds1, inds2, matched_ids[:, [1, 3]]


def _intersect2d_rows(ids1: np.ndarray, ids2: np.ndarray) -> np.ndarray:
    """intersect 2D array in row (axis=0) dimension"""
    intersections = []
    for item1 in ids1:
        for item2 in ids2:
            if np.array_equal(item1, item2):
                intersections.append(item1)
    intersections = np.array(intersections).astype(np.int64)
    return intersections


def bipartite_local_matching(pred_dets, dets):
    pred_dets_array = make_array_from_dets(pred_dets)
    dets_array = make_array_from_dets(dets)
    pred_inds1, inds1, _ = match_detections(pred_dets_array, dets_array)
    inds2, pred_inds2, _ = match_detections(dets_array, pred_dets_array)
    inds1 = np.vstack((pred_inds1, inds1)).T
    inds2 = np.vstack((pred_inds2, inds2)).T
    intersections = _intersect2d_rows(inds1, inds2)
    pred_inds = intersections[:, 0]
    inds = intersections[:, 1]
    return pred_inds, inds


def _get_tl_and_br(det: Detection) -> Tuple:
    return tl_br_from_cen_wh(det.x, det.y, det.w, det.h)


def hungarian_global_matching(dets1, dets2):
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - get_iou(_get_tl_and_br(det1), _get_tl_and_br(det2))
            loc_loss = np.linalg.norm([det2.x - det1.x, det2.y - det1.y])
            dist[i, j] = iou_loss + loc_loss
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def association_learning_matching(dets, dets2):
    # TODO bugs: there are several bugs
    # 1. time and frame_number therefore image crops are not correct
    # This can happen when track is stops for few frames
    # 2. One object match to different tracks. This is combined with
    # _make_new_track cause the track has the same id. e.g.
    # last track item track id 31 with the same object repeated twice.
    from copy import deepcopy

    import cv2
    import torch
    import torchvision
    from PIL import Image
    from sklearn.neighbors import KDTree

    from tracking import association_learning as al

    device = "cuda"
    model = al.AssociationNet(2048, 5).to(device)
    model.load_state_dict(
        torch.load("/home/fatemeh/Downloads/result_snellius/al/2_best.pth")["model"]
    )
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    crop_w, crop_h = 512, 256

    # TODO parameter
    step = 8
    vid_name = "437_cam12"
    image_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids/images")

    dets = make_array_from_dets(dets)
    dets2 = make_array_from_dets(dets2)
    frame_number = dets[0, 1]
    frame_number2 = dets2[0, 1]
    time = (frame_number2 - frame_number) / step

    im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:, :, ::-1]
    im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[
        :, :, ::-1
    ]

    kdt = KDTree(dets2[:, 7:9])
    _, inds = kdt.query(dets[:, 7:9], k=5)
    dets_inds = []
    dets2_inds = []
    for query_ind, ind in enumerate(inds):
        for _ in range(1):
            bbox1 = deepcopy(dets[query_ind, 2:7][None, :])
            jitter_x, jitter_y = np.random.normal(50, 10, 2)
            crop_x, crop_y = max(0, int(bbox1[0, 1] + jitter_x - crop_w / 2)), max(
                0, int(bbox1[0, 2] + jitter_y - crop_h / 2)
            )
            bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
            bboxes2 = deepcopy(dets2[ind, 2:7])
            bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y)

            bboxes2 = zero_out_of_image_bboxs(bboxes2, crop_w, crop_h)
            bbox1 = normalize_bboxs(bbox1, crop_w, crop_h)
            bboxes2 = normalize_bboxs(bboxes2, crop_w, crop_h)

            bbox1 = torch.tensor(bbox1[:, 1:]).unsqueeze(0).to(device).to(torch.float32)
            bboxes2 = (
                torch.tensor(bboxes2[:, 1:]).unsqueeze(0).to(device).to(torch.float32)
            )
            time_emb = torch.tensor([time / 4], dtype=torch.float32).to(device)

            imc = im[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            imc2 = im2[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            imc = np.ascontiguousarray(imc)
            imc2 = np.ascontiguousarray(imc2)
            imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
            imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)
            output = model(imt, bbox1, imt2, bboxes2, time_emb)
            argmax = torch.argmax(output, axis=1).item()

        dets_inds.append(query_ind)
        dets2_inds.append(ind[argmax])
    return np.array(dets_inds, dtype=np.int64), np.array(dets2_inds, dtype=np.int64)


def normalize_bboxs(bboxs, crop_w, crop_h):
    assert bboxs.shape[1] == 5
    bboxs = bboxs.astype(np.float32)
    return np.concatenate(
        (bboxs[:, 0:1], bboxs[:, 1:] / np.tile(np.array([crop_w, crop_h]), 2)), axis=1
    )


def change_center_bboxs(bboxs, crop_x, crop_y):
    assert bboxs.shape[1] == 5
    return np.concatenate(
        (bboxs[:, 0:1], bboxs[:, 1:] - np.tile(np.array([crop_x, crop_y]), 2)), axis=1
    )


def zero_out_of_image_bboxs(bboxes, crop_w, crop_h):
    assert bboxes.shape[1] == 5
    bboxs = bboxes.copy()
    bboxs[:, 1::2] = np.clip(bboxs[:, 1::2], 0, crop_w)
    bboxs[:, 2::2] = np.clip(bboxs[:, 2::2], 0, crop_h)
    bboxs[bboxs[:, 1] == bboxs[:, 3], 1:] = 0
    bboxs[bboxs[:, 2] == bboxs[:, 4], 1:] = 0
    return bboxs


def _connect_inds_to_detection_ids(dets):
    inds_to_det_ids = {i: det.det_id for i, det in enumerate(dets)}
    det_ids_to_inds = {det.det_id: id for i, det in enumerate(dets)}
    return inds_to_det_ids, det_ids_to_inds


def _make_new_track(det: Detection, new_track_id) -> Track:
    color = tuple(np.random.rand(3).astype(np.float16))
    det.track_id = new_track_id

    track = Track(
        dets=[det],
        color=color,
        status=Status.Untracked,
    )
    return track


def _initialize_track(dets):
    new_track_id = 0
    tracks = {}
    ids = range(len(dets))
    for id in ids:
        tracks[new_track_id] = _make_new_track(dets[id], new_track_id)
        new_track_id += 1
    return tracks, new_track_id


def initialize_tracks(
    det_folder: Path,
    filename_fixpart: str,
    width: int,
    height: int,
    first_frame: int = 0,
    format: str = "",
):
    det_path = det_folder / f"{filename_fixpart}_{first_frame + 1:{format}}.txt"
    dets = get_detections(det_path, width, height, first_frame)
    dets = get_cleaned_detections(det_path, width, height, first_frame)

    tracks, new_track_id = _initialize_track(dets)
    return tracks, new_track_id


def _track_predicted_unmatched(pred_dets, pred_inds, tracks):
    diff_ids = set(range(len(pred_dets))).difference(set(pred_inds))
    for id in diff_ids:
        current_track_id = pred_dets[id].track_id
        track = tracks[current_track_id]
        track.status = Status.Untracked
    return tracks


def _track_current_unmatched(dets, inds, frame_number, tracks, new_track_id):
    # TODO bug related to same detection assigned to different tracks
    # tracks dict keys are correct but in _make_new_track the det track_id is changed
    # although this is assigned to different track keys but their det object is changed in both tracks
    # so _get_predicted_locations returns same track_id
    diff_inds = set(range(len(dets))).difference(set(inds))
    for id in diff_inds:
        tracks[new_track_id] = _make_new_track(dets[id], new_track_id)
        new_track_id += 1
    return tracks, new_track_id


def _track_matches(
    pred_dets,
    dets,
    tracks,
    current_frame_number,
):
    # pred_inds, inds = association_learning_matching(pred_dets, dets)
    pred_inds, inds = hungarian_global_matching(pred_dets, dets)
    # pred_inds, inds = bipartite_local_matching(pred_dets, dets)

    if inds is None:
        return tracks, [], []

    unmatched_pred_inds = []
    unmatched_inds = []
    for pred_id, id in zip(pred_inds, inds):
        current_track_id = pred_dets[pred_id].track_id
        track = tracks[current_track_id]
        # kill tracks that are not tracked for a while
        if current_frame_number - track.dets[-1].frame_number > stopped_track_length:
            track.status = Status.Stoped
        else:
            dist = np.linalg.norm(
                [pred_dets[pred_id].x - dets[id].x, pred_dets[pred_id].y - dets[id].y]
            )
            if dist < accepted_flow_length:
                dets[id].track_id = current_track_id
                track.dets.append(dets[id])
                track.status = Status.Tracked

            else:
                unmatched_pred_inds.append(pred_id)
                unmatched_inds.append(id)
    matched_pred_inds = set(pred_inds).difference(set(unmatched_pred_inds))
    matched_inds = set(inds).difference(set(unmatched_inds))
    return tracks, matched_pred_inds, matched_inds


def _get_predicted_flow(track, current_frame_number, flow_decay_rate=0.5):
    flow = Point(0, 0)
    if len(track.dets) > 1:
        num_missing_frames = current_frame_number - track.dets[-1].frame_number
        diff_frame_number = track.dets[-1].frame_number - track.dets[-2].frame_number

        factor = num_missing_frames / diff_frame_number * flow_decay_rate
        x_coord_diff = track.dets[-1].x - track.dets[-2].x
        y_coord_diff = track.dets[-1].y - track.dets[-2].y
        flow = Point(
            x=np.int64(np.round(x_coord_diff * factor)),
            y=np.int64(np.round(y_coord_diff * factor)),
        )
    return flow


def _get_predicted_locations(tracks, current_frame_number):
    pred_dets = []
    for _, track in tracks.items():
        if track.status != Status.Stoped:
            pred_det = track.dets[-1]

            # flow = _get_predicted_flow(track, current_frame_number)
            # pred_loc = _update_det_loc_by_flow(pred_det, flow)

            pred_dets.append(pred_det)
            # if current_frame_number - pred_det.frame_number < min_track_length:
            #     pred_dets.append(pred_det)
            # else:
            #     track.status = Status.Stoped
    return pred_dets


def _remove_short_tracks(tracks):
    new_tracks = {}
    for track_id, track in tracks.items():
        if len(track.dets) > min_track_length:
            new_tracks[track_id] = _copy_track(track)
    return new_tracks


def _change_track_id(track: Track, new_id: int):
    for det in track.dets:
        det.track_id = new_id
    return track


def _reindex_tracks(tracks):
    new_tracks = {}
    for new_id, (_, track) in enumerate(tracks.items()):
        new_track = _copy_track(track)
        new_track = _change_track_id(new_track, new_id)
        new_tracks[new_id] = new_track
    return new_tracks


def compute_tracks(
    det_folder: Path,
    filename_fixpart: str,
    width: int,
    height: int,
    start_frame: int = 0,
    end_frame=None,
    step: int = 1,
    format: str = "",
):
    tracks, new_track_id = initialize_tracks(
        det_folder, filename_fixpart, width, height, start_frame, format
    )

    # start track
    # ===========
    for frame_number in tqdm(range(step, end_frame + 1, step)):
        # # track cleaning up
        # if frame_number % 20 == 0:
        #     tracks = _reindex_tracks(_remove_short_tracks(tracks))

        # ugly hack for yolo naming in yolo8, which is one based, e.g frame_1.txt
        if format == "":
            current_frame = frame_number + 1
        else:
            current_frame = frame_number
        det_path = det_folder / f"{filename_fixpart}_{current_frame:{format}}.txt"
        dets = get_detections(det_path, width, height, frame_number)
        dets = get_cleaned_detections(det_path, width, height, frame_number)
        pred_dets = _get_predicted_locations(tracks, frame_number)

        # track maches
        tracks, pred_inds, inds = _track_matches(
            pred_dets,
            dets,
            tracks,
            frame_number,
        )

        # unmatched tracks: predicted
        tracks = _track_predicted_unmatched(pred_dets, pred_inds, tracks)

        # unmatched tracks: current
        tracks, new_track_id = _track_current_unmatched(
            dets, inds, frame_number, tracks, new_track_id
        )
    return tracks


def hungarian_track(
    dets_path,
    filename_fixpart,
    width,
    height,
    start_frame,
    end_frame,
    step,
    format: str = "",
) -> np.ndarray:
    """
    Track objects in a video using the Hungarian algorithm.

    Parameters
    ----------
    dets_path : str
        Path to the directory containing detection files.
    filename_fixpart : str
        Fixed part of the filenames for detections.
    width : int
        Width of the video frames.
    height : int
        Height of the video frames.
    start_frame : int
        Starting frame number for tracking.
    end_frame : int
        Ending frame number for tracking.
    step : int
        Frame step size for processing.
    format : str, optional
        Format of the detection files. Default is "".

    Returns
    -------
    np.ndarray
        Array representing the tracks where each row corresponds to a detected object
        in a frame, and columns contain information about the object properties:
        tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st

    Notes
    -----
    This function uses the Hungarian algorithm to associate detections across frames
    and generate tracks.
    """

    tracks = compute_tracks(
        dets_path,
        filename_fixpart,
        width,
        height,
        start_frame,
        end_frame,
        step,
        format,
    )
    # tracks = _reindex_tracks(_remove_short_tracks(tracks))
    tracks = make_array_from_tracks(tracks)
    return tracks


def save_tracks_to_mot_format(
    save_file: Path, tracks: Union[np.ndarray, Dict[str, Track]], make_zip: bool = True
):
    """
    Save tracks in MOT format.

    Parameters
    ----------
    save_file : Path
        The zip file to save the MOT format files. e.g. /data/tmp.zip
    tracks : Union[np.ndarray, Dict[str, Track]]
        Either a NumPy array or a dictionary of tracks. If it's a NumPy array,
        it should have the format: (track_id, frame_number, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h).
        If it's a dictionary, it should have track IDs as keys and Track objects as values.
    make_zip : bool, optional
        Whether to create a zip archive of the saved files. Default is True.

    Notes
    -----
    MOT format is 1-based, including bbox. For more information, see: https://arxiv.org/abs/2003.09003
    The MOT format is as follows: frame_id, track_id, xtl, ytl, w, h, score, class, visibility
    The array format corresponds to: track_id, frame_number, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h
    NB. detections can be saved here. The detection IDs are used instead of track IDs.
    """
    tracks = deepcopy(tracks)

    track_folder = save_file.parent / "gt"
    track_folder.mkdir(parents=True, exist_ok=True)
    with open(track_folder / "labels.txt", "w") as wf:
        wf.write("fish")

    track_file = track_folder / "gt.txt"
    if isinstance(tracks, dict):
        with open(track_file, "w") as file:
            for track_id, track in tracks.items():
                for det in track.dets:
                    top_left_x = det.x - det.w / 2
                    top_left_y = det.y - det.h / 2
                    file.write(
                        f"{det.frame_number+1},{track_id+1},{top_left_x+1},{top_left_y+1},{det.w},{det.h},1,1,1.0\n"
                    )
    if isinstance(tracks, np.ndarray):
        with open(track_file, "w") as file:
            for item in tracks:
                if item[0] == -1:  # track_id=-1: detections. no track info.
                    item[0] = item[2]  # deepcopy it not
                if tracks.shape[1] == 12:
                    det_quality = int(item[11])
                else:
                    det_quality = 1
                file.write(
                    f"{int(item[1])+1},{int(item[0])+1},{item[3]+1},{item[4]+1},{int(item[9])},{int(item[10])},{det_quality},1,1.0\n"
                )
    if make_zip:
        shutil.make_archive(save_file.with_suffix(""), "zip", save_file.parent, "gt")
        shutil.rmtree(track_folder)


def load_tracks_from_mot_format(zip_file: Path) -> np.ndarray:
    """
    mot format: frame_id, track_id, xtl, ytl, w, h, score, class, visibility
    array format: track_id, frame_id, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h

    input:
        zip_file: either zip file containing track file or track file itself.
        zip file contains gt folder with gt.txt, label.txt. This file comes from
        cvat.
    """
    if zip_file.suffix == ".zip":
        shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
        track_file = zip_file.parent / zip_file.stem / "gt/gt.txt"
    else:
        track_file = zip_file

    tracks = []
    with open(track_file, "r") as file:
        for row in file:
            items = row.split("\n")[0].split(",")
            top_left_x, top_left_y, width, height = (
                float(items[2]) - 1,
                float(items[3]) - 1,
                float(items[4]),
                float(items[5]),
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
                int(items[6]),  # detection quality
            ]
            tracks.append(track)
    if zip_file.suffix == ".zip":
        shutil.rmtree(zip_file.parent / zip_file.stem)
    tracks = np.round(np.array(tracks)).astype(np.int64)
    det_qualities = tracks[:, -1]
    if set(det_qualities) == {1}:
        return tracks[:, :11]
    return tracks


def mot_to_cvat_xml(mot_file_path, video_name, width, height, keyframe_frequency=10):
    """
    Convert a MOT format file to CVAT XML for videos.

    Args:
    - mot_file_path (str): path to the input MOT format file.
    - video_name (str): name of the video.
    - width (int): width of the frames in the video.
    - height (int): height of the frames in the video.
    - keyframe_frequency (int, optional): frequency for keyframes. Default is 10.

    Returns:
    - str: path to the converted CVAT XML file.
    """

    # Read the MOT file and process detections
    detections = defaultdict(list)
    with open(mot_file_path, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            frame, track_id, x, y, w, h, _, _, _ = row
            frame = int(frame) - 1
            track_id = int(track_id) - 1
            detections[track_id].append(
                (frame, float(x), float(y), float(x) + float(w), float(y) + float(h))
            )

    # Create XML
    root = ET.Element("annotations")
    version = ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "unknown"
    ET.SubElement(task, "name").text = video_name
    ET.SubElement(task, "size").text = str(
        max(
            [int(frame) for detection in detections.values() for frame, *_ in detection]
        )
    )
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"
    ET.SubElement(task, "labels").append(ET.Element("label", name="fish"))
    ET.SubElement(task, "segments").append(
        ET.Element(
            "segment",
            id="0",
            start="0",
            stop=str(
                max(
                    [
                        int(frame)
                        for detection in detections.values()
                        for frame, *_ in detection
                    ]
                )
            ),
        )
    )
    ET.SubElement(task, "original_size").append(ET.Element("width", text=str(width)))
    ET.SubElement(task, "original_size").append(ET.Element("height", text=str(height)))

    for track_id, boxes in detections.items():
        track = ET.SubElement(root, "track", id=str(track_id), label="fish")
        for frame, xtl, ytl, xbr, ybr in boxes:
            ET.SubElement(
                track,
                "box",
                frame=str(frame),
                xtl=str(xtl),
                ytl=str(ytl),
                xbr=str(xbr),
                ybr=str(ybr),
                outside="0",
                occluded="0",
                keyframe="1"
                if frame % keyframe_frequency == 0
                or frame == boxes[0][0]
                or frame == boxes[-1][0]
                else "0",
            )

    tree = ET.ElementTree(root)
    output_file_path = mot_file_path.rsplit(".", 1)[0] + "_converted.xml"
    tree.write(output_file_path)

    return output_file_path


def cvat_xml_to_mot(cvat_xml_path, mot_file_path):
    """
    Convert a CVAT XML format file to MOT format.

    Args:
    - cvat_xml_path (str): path to the input CVAT XML file.
    - mot_file_path (str): path to the output MOT format file.

    Returns:
    - str: path to the converted MOT format file.
    """

    # Parse the XML file
    tree = ET.parse(cvat_xml_path)
    root = tree.getroot()

    # Extract track and box information
    mot_data = []
    for track in root.findall("track"):
        track_id = int(track.get("id")) + 1
        for box in track.findall("box"):
            frame = int(box.get("frame")) + 1
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            # Convert to MOT format (x, y, w, h)
            w = xbr - xtl
            h = ybr - ytl
            mot_data.append([frame, track_id, xtl, ytl, w, h, 1, 1, 1.0])

    # Write to MOT format file
    with open(mot_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        for row in mot_data:
            writer.writerow(row)

    return mot_file_path


def xml_to_mots(xml_path, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # since one of the tasks didn't have width and all videos have the same width I just get one of them.
    width = int(root.find(".//task/original_size/width").text) // 2

    task_id_to_name = {}
    for task_element in root.findall(".//task"):
        task_id = task_element.find("id").text
        task_name = task_element.find("name").text
        task_id_to_name[task_id] = task_name.split("_")[0]

    task_id_to_min_frame = {}
    for task_id in task_id_to_name.keys():
        frame_numbers = [
            int(box.get("frame"))
            for box in root.findall(f".//track[@task_id='{task_id}']/box")
        ]
        if frame_numbers:
            task_id_to_min_frame[task_id] = min(frame_numbers)

    task_id_to_min_track_id = {}
    for task_id in task_id_to_name.keys():
        track_ids = [
            int(box.get("id"))
            for box in root.findall(f".//track[@task_id='{task_id}']")
        ]
        if track_ids:
            task_id_to_min_track_id[task_id] = min(track_ids)

    for task_id, name in tqdm(task_id_to_name.items()):
        print(task_id, name)
        tracks1 = []
        tracks2 = []
        task_tracks = root.findall(f".//track[@task_id='{task_id}']")
        for task_track in task_tracks:
            track_id = int(task_track.get("id")) - task_id_to_min_track_id.get(
                task_id, 0
            )
            boxes = task_track.findall("box")
            for box in boxes:
                frame_number = int(box.get("frame")) - task_id_to_min_frame.get(
                    task_id, 0
                )
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                w = xbr - xtl
                h = ybr - ytl
                if xtl > width:
                    xtl -= width
                    xbr -= width
                    tracks2.append([frame_number, track_id, xtl, ytl, w, h])
                else:
                    tracks1.append([frame_number, track_id, xtl, ytl, w, h])
        _write_file(save_path / f"{name}_1.txt", tracks1)
        _write_file(save_path / f"{name}_2.txt", tracks2)


def _write_file(save_file, tracks):
    with open(save_file, "w") as wfile:
        for t in tracks:
            item = f"{t[0]+1},{t[1]+1},{t[2]:.2f},{t[3]:.2f},{t[4]:.2f},{t[5]:.2f},1,1,1.0\n"
            wfile.write(item)


def load_tracks_from_cvat_txt_format(track_file: Path) -> np.ndarray:
    """
    array format: track_id, frame_number, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h
    """
    tracks = np.round(np.loadtxt(track_file, skiprows=1, delimiter=",")).astype(
        np.int64
    )
    centers_x = np.int64(np.round((tracks[:, 5] + tracks[:, 3]) / 2)).reshape(-1, 1)
    centers_y = np.int64(np.round((tracks[:, 6] + tracks[:, 4]) / 2)).reshape(-1, 1)
    width = np.int64(np.round(tracks[:, 5] - tracks[:, 3])).reshape(-1, 1)
    height = np.int64(np.round(tracks[:, 6] - tracks[:, 4])).reshape(-1, 1)
    return np.concatenate((tracks, centers_x, centers_y, width, height), axis=1)


def save_tracks_to_cvat_txt_format(track_file: Path, tracks: np.ndarray):
    np.savetxt(
        track_file,
        tracks[:, :7],
        header="track_id,frame_number,det_id,xtl,ytl,xbr,ybr",
        delimiter=",",
        fmt="%d",
    )


def save_tracks(track_file: Path, tracks: Union[np.ndarray, Dict[str, Track]]):
    if isinstance(tracks, dict):
        with open(track_file, "w") as file:
            for track_id, track in tracks.items():
                for det in track.dets:
                    file.write(
                        f"{track_id},{det.frame_number},{det.det_id},{det.x},{det.y},{det.w},{det.h},{det.score:.2f},{track.status.value}\n"
                    )

    if isinstance(tracks, np.ndarray):
        with open(track_file, "w") as file:
            for track in tracks:
                file.write(
                    f"{track[0]},{track[1]},{track[2]},{track[3]},{track[4]},{track[5]},{track[6]},{track[7]:.2f},{track[8]}\n"
                )


def load_tracks(track_file):
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


def giou(bbox1, bbox2) -> float:
    # bbox1,2: (x_topleft, y_topleft, x_bottomright, y_bottomr1ight)

    def correct_bbox(bbox):
        if bbox[0] > bbox[2]:
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
        if bbox[1] > bbox[3]:
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
        return bbox

    bbox1 = correct_bbox(bbox1)
    bbox2 = correct_bbox(bbox2)

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right < x_left or y_bottom < y_top:
        return -1.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = intersection_area / float(area1 + area2 - intersection_area)

    convex_hull = abs(max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0])) * abs(
        max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1])
    )
    assert convex_hull != 0.0, f"{bbox1}, {bbox2}"
    giou = iou - (1 - intersection_area / float(convex_hull))
    assert giou >= -1.0, f"{bbox1}, {bbox2}"
    assert giou <= 1.0, f"{bbox1}, {bbox2}"
    return giou


def is_valid_bbox(bbox):
    """
    inputs:
        bbox1,2: tuple|list|np.ndarray
            (x_topleft, y_topleft, x_bottomright, y_bottomright)
    """

    if (bbox[0] == bbox[2]) | (bbox[1] == bbox[3]):
        return False
    else:
        return True


def get_iou(bbox1, bbox2) -> float:
    """
    Calculate IOU (Intersection over Union)

    inputs:
        bbox1,2: tuple|list|np.ndarray
            (x_topleft, y_topleft, x_bottomright, y_bottomright)
    """

    # If bbox is not a bbox.
    # TODO: maybe remove
    if (not is_valid_bbox(bbox1)) | (not is_valid_bbox(bbox2)):
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate IOU (Intersection over Union)
    intersect_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(
        0, min(y2_1, y2_2) - max(y1_1, y1_2)
    )
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = intersect_area / (area1 + area2 - intersect_area)
    return iou


def are_boxes_close(bbox1, bbox2, iou_thrs=0.5, dist_thrs=5):
    """
    Check if two bounding boxes are close to each other within a threshold.

    Parameters:
    - bbox1: Tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box.
    - bbox2: Tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box.
    - iou_thrs: Accepted Intersection over Union (IOU) threshold for intersection.
    - dist_thrs: Accepted distance threshold for closeness.

    Returns:
    - True if the bounding boxes intersect or are close to each other, False otherwise.
    """

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    iou1 = get_iou((x1_1, y1_1, x2_1 + dist_thrs, y2_1 + dist_thrs), bbox2)
    iou2 = get_iou(bbox1, (x1_2, y1_2, x2_2 + dist_thrs, y2_2 + dist_thrs))

    return iou1 > iou_thrs or iou2 > iou_thrs


def is_inside_bbox(bbox1, bbox2, threshold=0):
    """
    Check if bbox1 is inside bbox2 or within an accepted threshold.

    Parameters:
    - bbox1: Tuple | List | np.ndarray
        (x1, y1, x2, y2) representing the coordinates of the first bounding box.
    - bbox2: Tuple | List | np.ndarray
        (x1, y1, x2, y2) representing the coordinates of the second bounding box.
    - threshold: Accepted threshold for containment.

    Returns:
    - True if bbox1 is inside bbox2 or within the accepted threshold, False otherwise.
    """

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if bbox1 is inside bbox2
    is_inside = (x1_2 <= x1_1 <= x2_1 <= x2_2) and (y1_2 <= y1_1 <= y2_1 <= y2_2)

    # Check if bbox1 is within the accepted threshold in bbox2
    is_within_threshold = (
        x1_1 >= x1_2 - threshold
        and y1_1 >= y1_2 - threshold
        and x2_1 <= x2_2 + threshold
        and y2_1 <= y2_2 + threshold
    )

    return is_inside or is_within_threshold


def is_bbox_in_bbox(bbox1, bbox2, inters_thres=0.85) -> float:
    """
    inputs:
        bbox1,2: tuple|list|np.ndarray
            (x_topleft, y_topleft, x_bottomright, y_bottomright)
    """

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    inters_ratio1 = intersection_area / float(area1)
    inters_ratio2 = intersection_area / float(area2)
    if (inters_ratio1 >= inters_ratio2) & (inters_ratio1 > inters_thres):
        return True
    else:
        return False


def find_detection_in_track_by_frame_number(track, frame_number):
    for det in track.dets:
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
    return [det.frame_number for det in track.dets]


def find_track_id_by_coord_and_frame_number(tracks, x, y, frame_number, tolerance=3):
    for track_id, track in tracks.items():
        det = find_detection_in_track_by_frame_number(track, frame_number)
        if det:
            if (abs(det.x - x) < tolerance) & (abs(det.y - y) < tolerance):
                return track_id


def get_track_ind_from_track_id_frame_number(
    tracks: np.ndarray, track_id: int, frame_number: int
) -> np.ndarray:
    return np.where((tracks[:, 0] == track_id) & (tracks[:, 1] == frame_number))[0]


def get_track_from_track_id(tracks: np.ndarray, track_id: int) -> np.ndarray:
    # this is a copy
    return tracks[tracks[:, 0] == track_id]


def get_track_inds_from_track_id(tracks: np.ndarray, track_id: int) -> np.ndarray:
    return np.where(tracks[:, 0] == track_id)[0]


def _compute_disp_candidates(det1: Detection, dets2: Detection) -> List[int]:
    disp_candidates = []
    det_ids = []
    for det2 in dets2:
        disp = abs(det1.x - det2.x)
        rectification_error = abs(det1.y - det2.y)
        if (
            rectification_error
            < accepted_rect_error
            # and disp < largest_disparity
            # and disp > smallest_disparity
        ):
            disp_candidates.append(disp)
            det_ids.append(det2.det_id)
    return disp_candidates, det_ids


@dataclass
class Disparity:
    track_id: int
    frame_number: int
    det_id: int
    candidates: List[int]
    det_ids: List[int]


def get_detections_with_disparity(
    det_path_cam1,
    det_path_cam2,
    width: int,
    height: int,
) -> List[Disparity]:
    frame_number = int(det_path_cam1.stem.split("_")[-1]) - 1
    assert (
        frame_number == int(det_path_cam2.stem.split("_")[-1]) - 1
    ), "not a stereo pair"
    dets_cam1 = get_detections(det_path_cam1, width, height, frame_number)
    dets_cam2 = get_detections(det_path_cam2, width, height, frame_number)
    detections = []
    for det in dets_cam1:
        disp_candidates, det_ids = _compute_disp_candidates(det, dets_cam2)
        detection = Disparity(
            det.track_id, frame_number, det.det_id, disp_candidates, det_ids
        )
        detections.append(detection)
    return detections


def save_disparities(save_file: Path, disps: List[Disparity]):
    with open(save_file, "w") as wfile:
        wfile.write("track_id,frame_number,det_id,candidates,det_ids")
        for disp in disps:
            if len(disp.candidates) != 0:
                wfile.write("\n")
                wfile.write(
                    f"{disp.track_id};{disp.frame_number};{disp.det_id};{disp.candidates};{disp.det_ids}"
                )


def load_disparities(save_file) -> List[Disparity]:
    disparities = []
    with open(save_file, "r") as rfile:
        rfile.readline()
        for row in rfile:
            items = row.split("\n")[0].split(";")
            candidats = list(map(int, items[3].split("[")[1].split("]")[0].split(",")))
            det_ids = list(map(int, items[4].split("[")[1].split("]")[0].split(",")))
            disparity = Disparity(
                track_id=int(items[0]),
                frame_number=int(items[1]),
                det_id=int(items[2]),
                candidates=candidats,
                det_ids=det_ids,
            )
            disparities.append(disparity)
    return disparities


# TODO: broken
"""
def _assign_unique_disp(tracks, frame_number):
    for track_id, track in tracks.items():
        det = track.dets[-1]
        if (len(det.disp_candidates) == 1) and (frame_number == det.frame_number):
            disp = DispWithProb(
                det.disp_candidates[0],
                1.0,
                track.disp.count + 1,
                frame_number,
                det.frame_number,
            )
            track.disp = disp
        if frame_number != det.frame_number:
            track.disp.current_frame_number = frame_number
    return tracks
"""
