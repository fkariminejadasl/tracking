import sys
import tempfile
from pathlib import Path

import numpy as np

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.data_association import (get_iou, read_tracks_cvat_txt_format,
                                       save_tracks_cvat_txt_format)
from tracking.stats import (get_gt_object_match, make_array_from_tracks,
                            make_tracks_from_array)

data_path = Path(__file__).parent / "data"
annos = read_tracks_cvat_txt_format(data_path / "test_gt.txt")
atracks = read_tracks_cvat_txt_format(data_path / "test.txt")


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
    # (Detection(x=239.5, y=456.5, w=23, h=31, det_id=4, frame_number=-1, score=-1, camera_id=0),
    # Detection(x=239.0, y=456.0, w=26, h=36, det_id=5, frame_number=-1, score=-1, camera_id=0))

    det1, det2 = get_gt_object_match(
        atracks, annos, track_id=28, frame_number=32, thres=20, min_iou=0.05
    )
    assert det1[0] == 28
    assert det2[0] == 1
    # (Detection(x=783.0, y=434.5, w=44, h=21, det_id=28, frame_number=-1, score=-1, camera_id=0),
    # Detection(x=798.0, y=430.0, w=16, h=10, det_id=1, frame_number=-1, score=-1, camera_id=0))

    det1, det2 = get_gt_object_match(
        atracks, annos, track_id=28, frame_number=32, thres=20, min_iou=0.2
    )
    assert det1[0] == 28
    assert det2 == None
    # (Detection(x=783.0, y=434.5, w=44, h=21, det_id=28, frame_number=-1, score=-1, camera_id=0),
    # None)
