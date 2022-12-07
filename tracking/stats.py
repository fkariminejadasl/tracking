import numpy as np

from tracking.data_association import (Detection, Prediction, Status, Track,
                                       get_iou)


def _get_dets_from_indices_of_array(idxs, annos: np.ndarray):
    dets_anno = []
    for idx in idxs:
        anno = annos[idx]
        center_x = int(round((anno[3] + anno[5]) / 2))
        center_y = int(round((anno[4] + anno[6]) / 2))
        det = Detection(
            x=center_x,
            y=center_y,
            w=anno[5] - anno[3],
            h=anno[6] - anno[4],
            det_id=anno[0],
            frame_number=anno[1] + 1,  # my frame_number starts from 1
        )
        dets_anno.append(det)
    return dets_anno


def _get_dets_by_frame_number_from_array(annos: np.ndarray, frame_number: int):
    idxs = np.where(annos[:, 1] == frame_number)[0]
    assert len(idxs) != 0, f"frame {frame_number} is empty"
    return _get_dets_from_indices_of_array(idxs, annos)


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


def make_array_from_tracks(tracks) -> np.ndarray:
    tracks_array = []
    for track_id, track in tracks.items():
        for coord in track.coords:
            item = [
                track_id,
                coord.frame_number - 1,  # my frame_number starts from 1
                0,
                int(round(coord.x - coord.w / 2)),  # top left
                int(round(coord.y - coord.h / 2)),
                int(round(coord.x + coord.w / 2)),  # bottom right
                int(round(coord.y + coord.h / 2)),
                int(round(coord.x)),  # center
                int(round(coord.y)),
                coord.w,
                coord.h,
            ]
            tracks_array.append(item)
    return np.array(tracks_array).astype(np.int64)


def compare_with_gt_track(tracks_gt: np.ndarray, tracks: np.ndarray):
    # false positives should be calculated
    track_ids_gt = np.unique(tracks_gt[:, 0])
    track_id_gt = track_ids_gt[0]

    tp = fp = fn = tp_id = tn_id = fp_id = 0
    all_track_ids = []
    orig_track_id = -1
    idxs = np.where(tracks_gt[:, 0] == track_id_gt)[0]
    for idx in idxs:
        frame_number = tracks_gt[idx, 1]
        x_gt = tracks_gt[idx, 3]
        y_gt = tracks_gt[idx, 4]
        track_ids = _find_track_id_from_xyf_in_tracks_array(
            tracks, frame_number, x=x_gt, y=y_gt
        )
        if track_ids[0] != -1:
            # TODO some stats: thik about ids
            if orig_track_id == -1:
                orig_track_id = track_ids[0]
            if orig_track_id in track_ids:
                tp_id += 1
            else:
                fn_id += 1
            tp += 1
        else:
            fn += 1
    return tp, fp, fn, tp_id, tn_id, fp_id


def get_gt_object_match(atracks, annos, track_id, frame_number, thres=20, min_iou=0.1):
    # This function should replace _find_track_id_from_xyf_in_tracks_array.
    det_gt = annos[(annos[:, 0] == track_id) & (annos[:, 1] == frame_number)][0]

    candidates = atracks[
        (atracks[:, 1] == frame_number)
        & (
            (
                (abs(atracks[:, 3] - det_gt[3]) < thres)
                & (abs(atracks[:, 4] - det_gt[4]) < thres)
            )
            | (
                (abs(atracks[:, 5] - det_gt[5]) < thres)
                & (abs(atracks[:, 6] - det_gt[6]) < thres)
            )
        )
    ]
    if len(candidates) == 0:
        return det_gt, None
    ious = []
    dets = []
    for det in candidates:
        ious.append([det[0], get_iou(det_gt[3:7], det[3:7])])
        dets.append(det)
    ious = np.array(ious)
    iou_max = max(ious[:, 1])
    if iou_max < min_iou:
        return det_gt, None
    track_id = ious[ious[:, 1] == iou_max][0, 0]
    det = [det for det in dets if det[0] == track_id][0]
    return det_gt, det


def _find_track_id_from_xyf_in_tracks_array(tracks, frame_number, x_gt, y_gt, thres=4):
    candidate_idxs = np.where(
        (tracks[:, 1] == frame_number)
        & (abs(tracks[:, 3] - x_gt) < thres)
        & (abs(tracks[:, 4] - y_gt) < thres)
    )[0]
    return candidate_idxs
