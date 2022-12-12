import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import (
    get_iou,
    read_tracks_cvat_txt_format,
    save_tracks_cvat_txt_format,
)
from tracking.stats import (
    get_gt_object_match,
    get_stats_for_a_frame,
    get_stats_for_a_track,
    get_stats_for_tracks,
    make_array_from_tracks,
    make_tracks_from_array,
)

data_path = Path(__file__).parent / "data"
annos = read_tracks_cvat_txt_format(data_path / "tracks_gt.txt")
atracks = read_tracks_cvat_txt_format(data_path / "tracks.txt")


def test_get_iou():
    det1 = (0, 0, 4, 2)
    det2 = (2, 1, 3, 2)

    np.testing.assert_equal(get_iou(det1, det2), 0.125)

    det1 = (0, 0, 4, 2)
    det2 = (4, 2, 5, 6)
    np.testing.assert_equal(get_iou(det1, det2), 0.0)


def test_read_tracks_cvat_txt_format():
    tracks = make_tracks_from_array(atracks)
    tracks_array = make_array_from_tracks(tracks)

    with tempfile.NamedTemporaryFile() as tmp:
        track_file = tmp.name
        print(track_file)
        save_tracks_cvat_txt_format(Path(track_file), atracks)
        atracks_new = read_tracks_cvat_txt_format(Path(track_file))
        np.testing.assert_equal(atracks_new[800, :3], atracks[800, :3])
        np.testing.assert_almost_equal(
            atracks_new[800, 3:], atracks[800, 3:], decimal=0
        )

    np.testing.assert_equal(tracks_array[800, :3], atracks[800, :3])
    np.testing.assert_almost_equal(tracks_array[800, 3:], atracks[800, 3:], decimal=0)


def test_make_tracks_from_array_and_reverse():
    tracks = make_tracks_from_array(annos)
    tracks_array = make_array_from_tracks(tracks)

    np.testing.assert_equal(tracks_array[21567, :3], annos[21567, :3])
    np.testing.assert_almost_equal(tracks_array[21567, 3:], annos[21567, 3:], decimal=0)


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
    np.testing.assert_equal((tp, fp, fn, sw, uid), (371, 0, 228, 255, 13))
    assert tp + fn == len(annos[annos[:, 0] == gt_track_id])


@pytest.mark.slow
def test_get_stats_for_tracks():
    desired = np.loadtxt(
        data_path / "track_stats.txt", skiprows=1, delimiter=","
    ).astype(np.int64)
    stats = get_stats_for_tracks(annos, atracks)
    np.testing.assert_equal(stats, desired)
