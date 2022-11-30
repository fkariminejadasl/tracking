import sys
from pathlib import Path

import numpy as np

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import Detection, get_iou
from tracking.stats import make_array_from_tracks, make_tracks_from_array


def test_get_iou():
    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(2, 1, 3, 2, 1)

    np.testing.assert_almost_equal(get_iou(det1, det2), 0.167, decimal=2)

    det1 = Detection(0, 0, 4, 2, 0)
    det2 = Detection(4, 2, 2, 1, 1)
    np.testing.assert_almost_equal(get_iou(det1, det2), 0.0, decimal=2)


def test_make_tracks_from_array_and_reverse():
    data_path = Path(__file__).parent / "data/04_07_22_2_F_annotations.txt"

    annos = np.round(np.loadtxt(data_path, skiprows=1, delimiter=",")).astype(np.int64)
    tracks = make_tracks_from_array(annos)
    tracks_array = make_array_from_tracks(tracks)

    item_gen = tracks_array[21567]
    item_orig = annos[21567]
    assert item_orig[0] == item_orig[0]
    assert item_orig[1] == item_orig[1]
    assert abs(item_orig[3] - item_orig[3]) < 2
    assert abs(item_orig[4] - item_orig[4]) < 2
    assert item_orig[5] == item_orig[5]
    assert item_orig[6] == item_orig[6]
