import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import (
    bipartite_local_matching,
    clean_detections,
    compute_tracks,
    get_detections,
    get_detections_array,
    get_detections_with_disparity,
    get_iou,
    hungarian_global_matching,
    is_bbox_in_bbox,
    load_disparities,
    load_tracks_from_cvat_txt_format,
    load_tracks_from_mot_format,
    make_array_from_dets,
    make_array_from_tracks,
    make_dets_from_array,
    make_tracks_from_array,
    match_detections,
    save_tracks_to_cvat_txt_format,
    save_tracks_to_mot_format,
)
from tracking.stats import (
    get_gt_object_match,
    get_stats_for_frame,
    get_stats_for_track,
    get_stats_for_tracks,
)
from tracking.stereo_gt import get_matched_track_ids, load_matched_tracks_ids

data_path = Path(__file__).parent / "data"
annos = load_tracks_from_cvat_txt_format(data_path / "04_07_22_G_2_rect_valid_gt.txt")
atracks = load_tracks_from_cvat_txt_format(data_path / "tracks.txt")
im_width, im_height = 2098, 1220


def test_get_iou():
    det1 = (0, 0, 4, 2)
    det2 = (2, 1, 3, 2)

    np.testing.assert_equal(get_iou(det1, det2), 0.125)

    det1 = (0, 0, 4, 2)
    det2 = (4, 2, 5, 6)
    np.testing.assert_equal(get_iou(det1, det2), 0.0)


def test_is_bbox_in_bbox():
    adets = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    assert is_bbox_in_bbox(adets[5, 3:7], adets[1, 3:7]) == True
    assert is_bbox_in_bbox(adets[1, 3:7], adets[5, 3:7]) == False
    assert is_bbox_in_bbox(adets[1, 3:7], adets[8, 3:7]) == False


def test_load_tracks_from_cvat_txt_format():
    tracks = make_tracks_from_array(atracks)
    tracks_array = make_array_from_tracks(tracks)

    with tempfile.NamedTemporaryFile() as tmp:
        track_file = tmp.name
        save_tracks_to_cvat_txt_format(Path(track_file), atracks)
        atracks_new = load_tracks_from_cvat_txt_format(Path(track_file))
        np.testing.assert_equal(atracks_new[:, :2], atracks[:, :2])
        np.testing.assert_almost_equal(atracks_new[:, 3:], atracks[:, 3:], decimal=0)

    np.testing.assert_equal(tracks_array[:, :2], atracks[:, :2])
    np.testing.assert_almost_equal(tracks_array[:, 3:], atracks[:, 3:], decimal=0)


def test_load_tracks_from_mot_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_tracks_to_mot_format(Path(tmp_dir) / "tmp.txt", atracks, make_zip=False)
        atracks_new = load_tracks_from_mot_format(Path(tmp_dir) / "gt/gt.txt")
        np.testing.assert_equal(atracks_new[:, :2], atracks[:, :2])
        np.testing.assert_almost_equal(atracks_new[:, 3:], atracks[:, 3:], decimal=0)


def test_make_tracks_from_array_and_reverse():
    tracks = make_tracks_from_array(annos)
    tracks_array = make_array_from_tracks(tracks)

    np.testing.assert_equal(tracks_array[:, :2], annos[:, :2])
    np.testing.assert_almost_equal(tracks_array[:, 3:], annos[:, 3:], decimal=0)


def test_get_gt_object_match():
    det1, det2 = get_gt_object_match(
        atracks, annos, track_id=4, frame_number=0, thres=20
    )
    assert det1[0] == 4
    assert det2[0] == 5

    det1, det2 = get_gt_object_match(
        atracks, annos, track_id=28, frame_number=32, thres=20, min_iou=0.05
    )
    assert det1[0] == 28
    assert det2[0] == 1

    det1, det2 = get_gt_object_match(
        atracks, annos, track_id=28, frame_number=32, thres=20, min_iou=0.2
    )
    assert det1[0] == 28
    assert det2 == None


def test_get_gt_object_match_frame0():
    frame_number = 0
    gt_track_ids = np.unique(annos[annos[:, 1] == frame_number, 0])
    matched_ids = []
    for gt_track_id in gt_track_ids:
        det1, det2 = get_gt_object_match(
            atracks, annos, gt_track_id, frame_number, thres=20, min_iou=0.1
        )
        if det2 is not None:
            matched_ids.append([det1[0], det2[0]])
    matched_ids = np.array(matched_ids).astype(np.int64)

    desired = np.round(
        np.loadtxt(data_path / "matched_ids_frame0.txt", skiprows=1, delimiter=",")
    ).astype(np.int64)
    np.testing.assert_equal(matched_ids, desired)


def test_get_stats_for_frame():
    tp, fp, fn = get_stats_for_frame(annos, atracks, 0)
    np.testing.assert_equal((tp, fp, fn), (36, 5, 1))


def test_get_stats_for_track():
    gt_track_id = 7
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_track(annos, atracks, gt_track_id)
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (567, 0, 33, 0, 1))

    gt_track_id = 5
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_track(annos, atracks, gt_track_id)
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (419, 0, 181, 49, 3))

    gt_track_id = 28
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_track(annos, atracks, gt_track_id)
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (341, 0, 258, 193, 14))
    assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


def test_get_stats_for_track_after_iou_bug():
    atracks = load_tracks_from_cvat_txt_format(data_path / "tracks_iou_bug.txt")
    gt_track_id = 28
    tp, fp, fn, sw, uid, _ = get_stats_for_track(annos, atracks, gt_track_id)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (346, 0, 253, 199, 14))
    assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


def test_make_array_from_dets_reverse():
    dets = get_detections(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    dets_array = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    actual = make_array_from_dets(dets)
    np.testing.assert_equal(actual, dets_array)

    actual = make_dets_from_array(dets_array)
    assert actual[10].x == dets[10].x
    assert actual[10].y == dets[10].y
    assert actual[10].w == dets[10].w
    assert actual[10].h == dets[10].h
    assert actual[10].frame_number == dets[10].frame_number
    assert actual[10].det_id == dets[10].det_id


def test_clean_detections():
    dets_array = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    cleaned_dets = clean_detections(dets_array)
    removed_det_ids = [0, 2, 12, 25, 1, 5, 8, 10, 6]
    assert len(cleaned_dets) == len(dets_array) - len(removed_det_ids)
    for det_id in removed_det_ids:
        assert det_id not in list(cleaned_dets[:, 0])


def test_match_ddetections():
    desired = np.loadtxt(
        data_path / "matched_ids_dets.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    adets1 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    adets2 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_3.txt", im_width, im_height
    )
    _, _, matched_ids = match_detections(adets1, adets2)
    idxs1, idxs2, matched_ids_cleaned = match_detections(
        clean_detections(adets1), clean_detections(adets2)
    )
    assert len(matched_ids) == 39
    np.testing.assert_equal(matched_ids_cleaned, desired)
    assert matched_ids_cleaned[0, 0] == clean_detections(adets1)[idxs1[0], 2]
    assert matched_ids_cleaned[0, 1] == clean_detections(adets2)[idxs2[0], 2]


def test_match_ddetections2():
    adets1 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_1.txt", im_width, im_height
    )
    adets2 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height
    )
    _, _, matched_ids = match_detections(adets1, adets2)
    idxs1, idxs2, matched_ids_cleaned = match_detections(
        clean_detections(adets1), clean_detections(adets2)
    )
    assert matched_ids_cleaned[0, 0] == clean_detections(adets1)[idxs1[0], 2]
    assert matched_ids_cleaned[0, 1] == clean_detections(adets2)[idxs2[0], 2]
    assert matched_ids_cleaned[-1, 0] == clean_detections(adets1)[idxs1[-1], 2]
    assert matched_ids_cleaned[-1, 1] == clean_detections(adets2)[idxs2[-1], 2]


def test_bipartite_matching():
    pred_dets = load_tracks_from_cvat_txt_format(data_path / "pred_dets_56.txt")
    dets = load_tracks_from_cvat_txt_format(data_path / "dets_56.txt")
    pred_ids, ids = bipartite_local_matching(
        make_dets_from_array(pred_dets), make_dets_from_array(dets)
    )
    desired = np.loadtxt(
        data_path / "matched_ids_56.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    np.testing.assert_equal(pred_ids, desired[:, 0])
    np.testing.assert_equal(ids, desired[:, 1])


def test_hungarian_global_matching():
    pred_dets = load_tracks_from_cvat_txt_format(data_path / "pred_dets_56.txt")
    dets = load_tracks_from_cvat_txt_format(data_path / "dets_56.txt")
    pred_ids, ids = hungarian_global_matching(
        make_dets_from_array(pred_dets), make_dets_from_array(dets)
    )
    desired = np.loadtxt(
        data_path / "matched_ids_56.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    np.testing.assert_equal(pred_ids, desired[:, 0])
    np.testing.assert_equal(ids, desired[:, 1])


@pytest.mark.slow
def test_get_stats_for_tracks():
    desired = np.loadtxt(
        data_path / "track_stats.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    track_stats = get_stats_for_tracks(annos, atracks)
    np.testing.assert_equal(track_stats, desired)
    for gt_track_id in range(40):
        tp = track_stats[gt_track_id, 1]
        fn = track_stats[gt_track_id, 3]
        assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.temp
def test_get_stats_for_track_after_no_prediction():
    atracks = load_tracks_from_cvat_txt_format(data_path / "tracks_no_prediction.txt")
    track_stats = get_stats_for_tracks(annos, atracks)
    desired = np.loadtxt(
        data_path / "track_stats_no_prediction.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    np.testing.assert_equal(track_stats, desired)
    for gt_track_id in range(40):
        tp = track_stats[gt_track_id, 1]
        fn = track_stats[gt_track_id, 3]
        assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.temp
def test_get_stats_for_track_after_new_tracks():
    atracks = load_tracks_from_cvat_txt_format(data_path / "tracks_new_tracks.txt")
    track_stats = get_stats_for_tracks(annos, atracks)
    desired = np.loadtxt(
        data_path / "track_stats_new_tracks.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    np.testing.assert_equal(track_stats, desired)
    for gt_track_id in range(40):
        tp = track_stats[gt_track_id, 1]
        fn = track_stats[gt_track_id, 3]
        assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.temp
def test_get_stats_for_track_after_remove_occlusions():
    atracks = load_tracks_from_cvat_txt_format(
        data_path / "tracks_remove_occlusions.txt"
    )
    track_stats = get_stats_for_tracks(annos, atracks)
    desired = np.loadtxt(
        data_path / "track_stats_remove_occlusions.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    np.testing.assert_equal(track_stats, desired)
    for gt_track_id in range(40):
        tp = track_stats[gt_track_id, 1]
        fn = track_stats[gt_track_id, 3]
        assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.failed
def test_compute_track():
    tracks = compute_tracks(
        data_path, "04_07_22_G_2_rect_valid", 2, im_width, im_height, 3
    )
    tracks_array = make_array_from_tracks(tracks)

    atracks = np.loadtxt(
        data_path / "tracks_iou_bug.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    desired = atracks[atracks[:, 1] < 3]

    np.testing.assert_equal(tracks_array[:, :7], desired)


@pytest.mark.temp
def test_disparities():
    desired = load_disparities(data_path / "disparities_frame81.txt")
    disparities = get_detections_with_disparity(
        data_path / "04_07_22_F_2_rect_valid_82.txt",
        data_path / "04_07_22_G_2_rect_valid_82.txt",
        im_width,
        im_height,
    )
    actual = []
    for disparity in disparities:
        if len(disparity.candidates) != 0:
            actual.append(disparity)
    for disp_act, disp_desired in zip(actual, desired):
        assert disp_act.track_id == disp_desired.track_id
        assert disp_act.frame_number == disp_desired.frame_number
        assert disp_act.det_id == disp_desired.det_id
        assert disp_act.candidates == disp_desired.candidates
        assert disp_act.det_ids == disp_desired.det_ids


@pytest.mark.slow
def test_get_matched_track_ids():
    desired = np.array(load_matched_tracks_ids())

    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    matches = np.array(get_matched_track_ids(annos1, annos))
    matches = matches[matches[:, 2] < 5]
    np.testing.assert_equal(matches[:, :2], desired[:, :2])


from tracking.data_association import (
    cen_wh_from_tl_br,
    get_track_from_track_id,
    interpolate_two_bboxes,
    tl_br_from_cen_wh,
)
from tracking.stereo_gt import get_disparity_info_from_stereo_track
from tracking.tracklet_operations import (
    add_remove_tracks,
    get_start_ends_missing_frames,
)


# TODO maybe remove
def test_add_remove_tracks():
    remove_tracks = np.random.randint(10, size=(3, 2))
    remove_lengths = np.random.randint(10, size=(3, 2))
    add_tracks = np.empty(shape=(0, 2), dtype=np.int64)

    desired = remove_tracks.copy()
    inds = np.array([0])
    while remove_tracks.size != 0:
        remove_tracks, remove_lengths, add_tracks = add_remove_tracks(
            remove_tracks, remove_lengths, add_tracks, inds
        )
    np.testing.assert_equal(add_tracks, desired)
    assert remove_tracks.size == 0
    assert remove_lengths.size == 0


def test_get_disparity_info_from_stereo_track():
    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    annos2 = annos.copy()

    len_disparity = 20
    track1_id = 21
    track2_id = 16
    track1 = get_track_from_track_id(annos1, track1_id)
    track2 = get_track_from_track_id(annos2, track2_id)
    disparity_info = get_disparity_info_from_stereo_track(
        track1[: 80 + len_disparity], track2[80:150]
    )
    align_errors = disparity_info[:, 3]
    desired = np.array([1, 1, 0, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 1])
    assert disparity_info.shape[0] == len_disparity
    np.testing.assert_equal(align_errors, desired)
    np.testing.assert_equal(disparity_info[:, 0], np.repeat(track1_id, len_disparity))
    np.testing.assert_equal(disparity_info[:, 1], np.repeat(track2_id, len_disparity))
    np.testing.assert_equal(disparity_info[:, 2], np.arange(80, 100))


def test_interpolate_two_bboxes():
    bbox1 = 2, 2, 5, 5
    bbox2 = 22, 32, 15, 25
    frame_number1 = 20
    frame_number2 = 24
    bboxes = interpolate_two_bboxes(bbox1, bbox2, frame_number1, frame_number2)
    desired = [
        (7.0, 9.5, 7.5, 10.0),
        (12.0, 17.0, 10.0, 15.0),
        (17.0, 24.5, 12.5, 20.0),
    ]
    np.testing.assert_equal(bboxes, desired)


def test_get_start_ends_missing_frames():
    missing_frames = np.array([4, 5, 6, 10, 11, 15, 18, 19])
    desired = [(3, 7), (9, 12), (14, 16), (17, 20)]

    start_ends = get_start_ends_missing_frames(missing_frames)
    np.testing.assert_equal(start_ends, desired)

    missing_frames = np.append(missing_frames, 25)
    desired = desired + [(24, 26)]

    start_ends = get_start_ends_missing_frames(missing_frames)
    np.testing.assert_equal(start_ends, desired)


def test_tl_br_from_cen_wh_and_reverse():
    det = np.array([43, 599, 0, 430, 440, 466, 461, 448, 450, 36, 21])
    cen_wh = cen_wh_from_tl_br(*det[3:7])
    tl_br = tl_br_from_cen_wh(*det[7:])

    np.testing.assert_almost_equal(cen_wh, det[7:11], decimal=0)
    np.testing.assert_almost_equal(tl_br, det[3:7], decimal=0)


"""
a = np.random.randint(10, size=(3, 2))
b = np.empty(shape=(0, 2), dtype=np.int64)
while a.size != 0:
    inds = np.array([0])
    b = np.append(b, a[inds], axis=0)
    a = np.delete(a, inds, axis=0)


# S = {s_i_f}, LS = {ls_i_f} -> TS
# P = {p_i_f}, LP = {lp_i_f} -> TP

# init: calculate length, sort (S,P)
# some for loop (P_tracks_ids, S_tracks_ids)
#   if (LP[0] > SP[0]):
#       match(p_track, tracks2) -> s_track1, s_track2, ..., s_trackn (s_inds)
#       add with new_track_id + remove + update lengths + sot (S, P, LS, LP -> TS, TP)
# remaining add to TS, TP (-1 not matched)

# think about align_errors (some drawing)
# stop criteria (S, P not empty)


# annos1 = load_tracks_from_cvat_txt_format(
#         data_path / "04_07_22_F_2_rect_valid_gt.txt"
#     )
# annos2 = load_tracks_from_cvat_txt_format(
#         data_path / "04_07_22_G_2_rect_valid_gt.txt"
#     )

track1 = da.get_track_from_track_id(annos1, 21)
track2 = da.get_track_from_track_id(annos2, 16)

p_track1 = da.get_track_from_track_id(tracks1, 17)
p_track2 = da.get_track_from_track_id(tracks1, 72)
s_track1 = da.get_track_from_track_id(tracks2, 14)
s_track2 = da.get_track_from_track_id(tracks2, 79)

# 7 <-> 24
# 7  -> 5, 54, 71, 79, 84, 90 len:(80, 107, 39, 12, 58, 201)
# 24 -> 21, 63  len:(262, 313)


# len(s_track1), len(s_track2), len(p_track1), len(p_track2): 361, 219, 236, 332
# tk.match_primary_track_to_secondry_tracklets(p_track1, tracks2)
# [[14, 1.0], [38, 6.0], [39, 3.0], [48, 2.0], [50, 1.0], [56, 6.0]]
"""
