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
    # # array format: track_id, frame_id, outside, xtl, ytl, xbr, ybr, xc, yc, w, h
    tracks_array = []
    for track_id, track in tracks.items():
        for det in track.coords:
            item = [
                track_id,
                det.frame_number - 1,  # my frame_number starts from 1
                0,
                int(round(det.x - det.w / 2)),  # top left
                int(round(det.y - det.h / 2)),
                int(round(det.x + det.w / 2)),  # bottom right
                int(round(det.y + det.h / 2)),
                int(round(det.x)),  # center
                int(round(det.y)),
                det.w,
                det.h,
            ]
            tracks_array.append(item)
    return np.array(tracks_array).astype(np.int64)


def get_gt_object_match(atracks, annos, track_id, frame_number, thres=20, min_iou=0.1):
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


def get_stats_for_a_frame(annos, atracks, frame_number):
    tp = fp = fn = 0
    gt_track_ids = np.unique(annos[annos[:, 1] == frame_number, 0])
    matched_ids = []
    for gt_track_id in gt_track_ids:
        det1, det2 = get_gt_object_match(
            atracks, annos, gt_track_id, frame_number, thres=20, min_iou=0.1
        )
        if det2 is None:
            fn += 1
        else:
            tp += 1
            matched_ids.append([det1[0], det2[0]])
    matched_ids = np.array(matched_ids).astype(np.int64)
    track_ids = np.unique(atracks[atracks[:, 1] == frame_number, 0])
    diff_ids = set(track_ids).difference(set(matched_ids[:, 1]))
    fp = len(diff_ids)

    # gt_diff_ids = set(gt_track_ids).difference(set(matched_ids[:, 0]))
    # print(f"matched ids tracks:\n{matched_ids}")
    # print(f"diff ids tracks:\n{diff_ids}")
    # print(f"diff ids gt:\n{gt_diff_ids}")
    return tp, fp, fn


def get_stats_for_a_track(annos, atracks, track_id):
    tp = fp = fn = 0
    frame_numbers = annos[annos[:, 0] == track_id, 1]
    matched_ids = []
    for frame_number in frame_numbers:
        det1, det2 = get_gt_object_match(
            atracks, annos, track_id, frame_number, thres=20, min_iou=0.1
        )
        if det2 is None:
            fn += 1
        else:
            tp += 1
            matched_ids.append([det1[0], det2[0], frame_number])
    matched_ids = np.array(matched_ids).astype(np.int64)

    unique_ids = np.unique(np.sort(matched_ids[:, 1]))
    freq, _ = np.histogram(
        matched_ids[:, 1], bins=np.hstack((unique_ids, unique_ids[-1] + 1))
    )
    main_track_id = unique_ids[freq == max(freq)][0]
    no_switch_ids = len(matched_ids[matched_ids[:, 1] != main_track_id, 1])
    no_unique_ids = len(unique_ids)

    # here fn is calculated based on dominant track_id.
    main_track_frame_numbers = atracks[atracks[:, 0] == main_track_id, 1]
    matched_main_track_frame_numbers = matched_ids[
        matched_ids[:, 1] == main_track_id, 2
    ]
    fp = len(set(main_track_frame_numbers).difference(matched_main_track_frame_numbers))
    return tp, fp, fn, no_switch_ids, no_unique_ids, matched_ids


def get_stats_for_tracks(annos, atracks):
    stats = []
    for track_id in np.unique(annos[:, 0]):
        tp, fp, fn, sw, uid, _ = get_stats_for_a_track(annos, atracks, track_id)
        stats.append([track_id, tp, fp, fn, sw, uid])
    return np.array(stats).astype(np.int64)
