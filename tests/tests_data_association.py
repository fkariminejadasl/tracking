import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import (
    are_boxes_close,
    bipartite_local_matching,
    cen_wh_from_tl_br,
    clean_detections,
    get_detections,
    get_detections_array,
    get_detections_with_disparity,
    get_iou,
    get_track_from_track_id,
    hungarian_global_matching,
    is_bbox_in_bbox,
    is_inside_bbox,
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
    tl_br_from_cen_wh,
    zero_out_of_image_bboxs,
)
from tracking.postprocess import (
    get_start_ends_missing_frames,
    interpolate_two_bboxes,
    reindex_tracks,
    remove_short_tracks,
)
from tracking.stats import (
    get_gt_object_match,
    get_stats_for_frame,
    get_stats_for_track,
    get_stats_for_tracks,
)

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


def test_are_boxes_close():
    kwargs = dict(iou_thrs=0, dist_thrs=2)
    bbox1 = (2, 2, 6, 6)
    bbox2 = (7, 5, 10, 6)
    result = are_boxes_close(bbox1, bbox2, **kwargs)
    assert result == True
    bbox1 = (2, 2, 6, 6)
    bbox2 = (7, 7, 10, 10)
    result = are_boxes_close(bbox1, bbox2, **kwargs)
    assert result == True
    result = are_boxes_close(bbox2, bbox1, **kwargs)
    assert result == True
    bbox2 = (2, 7, 6, 10)
    result = are_boxes_close(bbox1, bbox2, **kwargs)
    assert result == True
    result = are_boxes_close(bbox2, bbox1, **kwargs)
    assert result == True
    bbox2 = (7, 2, 10, 6)
    result = are_boxes_close(bbox1, bbox2, **kwargs)
    assert result == True
    result = are_boxes_close(bbox2, bbox1, **kwargs)
    assert result == True
    kwargs = dict(iou_thrs=0, dist_thrs=0)
    result = are_boxes_close(bbox2, bbox1, **kwargs)
    assert result == False


def test_is_bbox_in_bbox():
    adets = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
    )
    assert is_bbox_in_bbox(adets[5, 3:7], adets[1, 3:7]) == True
    assert is_bbox_in_bbox(adets[1, 3:7], adets[5, 3:7]) == False
    assert is_bbox_in_bbox(adets[1, 3:7], adets[8, 3:7]) == False


def test_is_inside_bbox():
    bbox1 = (2, 2, 6, 6)
    bbox2 = (1, 1, 5, 5)
    assert is_inside_bbox(bbox1, bbox2, threshold=1) == True
    assert is_inside_bbox(bbox1, bbox2, threshold=0) == False


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
        save_tracks_to_mot_format(Path(tmp_dir) / "tmp.zip", atracks, make_zip=False)
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
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
    )
    dets_array = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
    )
    actual = make_array_from_dets(dets)
    np.testing.assert_equal(actual, dets_array[:, :11])

    actual = make_dets_from_array(dets_array)
    assert actual[10].x == dets[10].x
    assert actual[10].y == dets[10].y
    assert actual[10].w == dets[10].w
    assert actual[10].h == dets[10].h
    assert actual[10].frame_number == dets[10].frame_number
    assert actual[10].det_id == dets[10].det_id


def test_clean_detections():
    dets_array = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
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
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
    )
    adets2 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_3.txt", im_width, im_height, 2
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
        data_path / "04_07_22_G_2_rect_valid_1.txt", im_width, im_height, 0
    )
    adets2 = get_detections_array(
        data_path / "04_07_22_G_2_rect_valid_2.txt", im_width, im_height, 1
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


def test_zero_out_of_image_bboxs():
    bboxs = np.array(
        [
            [9, 205, -1, 216, 27],
            [10, 489, 30, 501, 49],
            [11, 406, 427, 417, 434],
            [5, 416, 548, 453, 570],
            [6, 541, 515, 558, 529],
        ]
    )
    desired = np.array(
        [
            [9, 205, 0, 216, 27],
            [10, 489, 30, 501, 49],
            [11, 0, 0, 0, 0],
            [5, 0, 0, 0, 0],
            [6, 0, 0, 0, 0],
        ]
    )
    actual = zero_out_of_image_bboxs(bboxs, 512, 256)
    np.testing.assert_equal(actual, desired)
