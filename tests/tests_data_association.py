import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import (
    clean_detections,
    compute_tracks,
    get_detections,
    get_detections_array,
    get_iou,
    is_bbox_in_bbox,
    make_array_from_dets,
    make_array_from_tracks,
    make_dets_from_array,
    make_tracks_from_array,
    match_detections,
    read_tracks_cvat_txt_format,
    read_tracks_from_mot_format,
    save_tracks_cvat_txt_format,
    save_tracks_to_mot_format,
)
from tracking.stats import (
    get_gt_object_match,
    get_stats_for_a_frame,
    get_stats_for_a_track,
    get_stats_for_tracks,
)

data_path = Path(__file__).parent / "data"
annos = read_tracks_cvat_txt_format(data_path / "tracks_gt.txt")
atracks = read_tracks_cvat_txt_format(data_path / "tracks.txt")
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


def test_read_tracks_cvat_txt_format():
    tracks = make_tracks_from_array(atracks)
    tracks_array = make_array_from_tracks(tracks)

    with tempfile.NamedTemporaryFile() as tmp:
        track_file = tmp.name
        save_tracks_cvat_txt_format(Path(track_file), atracks)
        atracks_new = read_tracks_cvat_txt_format(Path(track_file))
        np.testing.assert_equal(atracks_new[:, :2], atracks[:, :2])
        np.testing.assert_almost_equal(atracks_new[:, 3:], atracks[:, 3:], decimal=0)

    np.testing.assert_equal(tracks_array[:, :2], atracks[:, :2])
    np.testing.assert_almost_equal(tracks_array[:, 3:], atracks[:, 3:], decimal=0)


def test_read_tracks_from_mot_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_tracks_to_mot_format(Path(tmp_dir) / "tmp.txt", atracks, make_zip=False)
        atracks_new = read_tracks_from_mot_format(Path(tmp_dir) / "gt/gt.txt")
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


def test_get_stats_for_a_frame():
    tp, fp, fn = get_stats_for_a_frame(annos, atracks, 0)
    np.testing.assert_equal((tp, fp, fn), (36, 5, 1))


def test_get_stats_for_a_track():
    gt_track_id = 7
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_a_track(
        annos, atracks, gt_track_id
    )
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (567, 0, 33, 0, 1))

    gt_track_id = 5
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_a_track(
        annos, atracks, gt_track_id
    )
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (419, 0, 181, 49, 3))

    gt_track_id = 28
    desired = np.loadtxt(
        data_path / f"matched_ids_track{gt_track_id}.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    tp, fp, fn, sw, uid, matched_ids = get_stats_for_a_track(
        annos, atracks, gt_track_id
    )
    np.testing.assert_equal(matched_ids, desired)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (341, 0, 258, 193, 14))
    assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


def test_get_stats_for_a_track_after_iou_bug():
    atracks = read_tracks_cvat_txt_format(data_path / "tracks_iou_bug.txt")
    gt_track_id = 28
    tp, fp, fn, sw, uid, _ = get_stats_for_a_track(annos, atracks, gt_track_id)
    np.testing.assert_equal((tp, fp, fn, sw, uid), (346, 0, 253, 199, 14))
    assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.slow
def test_get_stats_for_tracks():
    desired = np.loadtxt(
        data_path / "track_stats.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    stats = get_stats_for_tracks(annos, atracks)
    np.testing.assert_equal(stats, desired)


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
    assert len(cleaned_dets) == len(dets_array) - 2
    assert 0 not in list(cleaned_dets[:, 0])
    assert 1 not in list(cleaned_dets[:, 0])


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
    assert len(matched_ids) == 33
    np.testing.assert_equal(matched_ids_cleaned, desired)
    assert matched_ids_cleaned[0, 0] == clean_detections(adets1)[idxs1[0], 0]
    assert matched_ids_cleaned[0, 1] == clean_detections(adets2)[idxs2[0], 0]


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
