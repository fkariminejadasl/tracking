import sys
from pathlib import Path

import numpy as np

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import Detection, get_iou
from tracking import stats

data_path = Path(__file__).parent / "data"
annos = np.round(
    np.loadtxt(data_path / "test_gt.txt", skiprows=1, delimiter=",")
).astype(np.int64)
atracks = np.round(
    np.loadtxt(data_path / "test.txt", skiprows=1, delimiter=",")
).astype(np.int64)


def test_get_iou():
    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(2, 1, 3, 2, 1)

    np.testing.assert_almost_equal(get_iou(det1, det2), 0.167, decimal=2)

    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(4, 2, 2, 1, 1)
    np.testing.assert_almost_equal(get_iou(det1, det2), 0.0, decimal=2)


def test_make_tracks_from_array_and_reverse():
    tracks = stats.make_tracks_from_array(annos)
    tracks_array = stats.make_array_from_tracks(tracks)

    item_gen = tracks_array[21567]
    item_orig = annos[21567]
    assert item_orig[0] == item_orig[0]
    assert item_orig[1] == item_orig[1]
    assert abs(item_orig[3] - item_orig[3]) < 2
    assert abs(item_orig[4] - item_orig[4]) < 2
    assert item_orig[5] == item_orig[5]
    assert item_orig[6] == item_orig[6]


def test_get_gt_object_match():
    det1, det2 = stats.get_gt_object_match(
        atracks, annos, track_id=4, frame_number=0, thres=20
    )
    assert det1.det_id == 4
    assert det2.det_id == 5
    # (Detection(x=239.5, y=456.5, w=23, h=31, det_id=4, frame_number=-1, score=-1, camera_id=0),
    # Detection(x=239.0, y=456.0, w=26, h=36, det_id=5, frame_number=-1, score=-1, camera_id=0))

    det1, det2 = stats.get_gt_object_match(
        atracks, annos, track_id=28, frame_number=32, thres=20, min_iou=0.05
    )
    assert det1.det_id == 28
    assert det2.det_id == 1
    # (Detection(x=783.0, y=434.5, w=44, h=21, det_id=28, frame_number=-1, score=-1, camera_id=0),
    # Detection(x=798.0, y=430.0, w=16, h=10, det_id=1, frame_number=-1, score=-1, camera_id=0))

    det1, det2 = stats.get_gt_object_match(
        atracks,
        annos,
        track_id=28,
        frame_number=32,
        thres=20,
    )
    assert det1.det_id == 28
    assert det2 == None
    # (Detection(x=783.0, y=434.5, w=44, h=21, det_id=28, frame_number=-1, score=-1, camera_id=0),
    # None)
