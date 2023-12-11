from pathlib import Path

import cv2
import numpy as np

from tracking import data_association as da
from tracking import multi_stage_tracking as ms

np.random.seed(1000)

kwargs = ms.get_model_args()
vid_name, frame_number1, step, folder = 6, 16, 8, "240hz"
main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
image_path = main_path / "images"
frame_number2 = frame_number1 + step

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]
# missuse tracks for detections
dets1[:, 2] = dets1[:, 0]
dets2[:, 2] = dets2[:, 0]


image1 = cv2.imread(str(image_path / f"{vid_name}_frame_{frame_number1:06d}.jpg"))
image2 = cv2.imread(str(image_path / f"{vid_name}_frame_{frame_number2:06d}.jpg"))
features1 = ms.calculate_deep_features(dets1[:, 2], dets1, image1, **kwargs)
features2 = ms.calculate_deep_features(dets2[:, 2], dets2, image2, **kwargs)


def test_merge_intersecting_lists():
    occluded = [[8, 10], [7, 9], [6, 8, 10]]
    expected = [[6, 8, 10], [7, 9]]
    new = ms.merge_intersecting_lists(occluded)
    assert new == expected


def test_get_occluded_dets():
    bboxes = [
        [0, 0, 3, 3],
        [2, 2, 5, 5],
        [7, 7, 9, 9],
        [4, 4, 6, 6],
        [1, 1, 2, 2],
        [8, 8, 10, 10],
    ]
    bboxes = np.array(bboxes).astype(np.int64)
    other_columns = np.repeat(np.arange(len(bboxes))[None], 3, axis=0).T
    dets = np.concatenate((other_columns, bboxes), axis=1)
    groups = ms.get_occluded_dets(dets, close_iou_thrs=0, close_dist_thrs=0)
    assert groups == [[0, 1, 3, 4], [2, 5]]


def test_merge_overlapping_keys():
    input = {
        (4, 6): (0, 8, 6),
        (3,): (1, 3),
        (4,): (8, 6),
        (9, 10): (11, 12, 13),
        (1, 4): (0, 8),
        (2, 3): (1, 2, 3),
    }
    expected = {(1, 4, 6): (0, 6, 8), (2, 3): (1, 2, 3), (9, 10): (11, 12, 13)}
    result = ms.merge_overlapping_keys(input)
    assert result == expected


def test_merge_overlapping_keys_and_values():
    input = {(6, 7): (1,), (6,): (2, 3), (4,): (4,), (9, 8): (5, 1)}
    expected = {(6, 7, 8, 9): (1, 2, 3, 5), (4,): (4,)}
    result = ms.merge_overlapping_keys_and_values(input)
    assert result == expected


def test_find_match_groups():
    exp_occluded1 = [[6, 7], [13, 17], [21, 29]]
    exp_occluded2 = [[13, 17], [21, 29]]
    flatten = [v for vv in exp_occluded1 + exp_occluded2 for v in vv]
    exp_n_occluded = set(range(31)).difference(flatten)
    # {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31}
    exp_matching_groups = {(6, 7): (6, 7), (13, 17): (13, 17), (21, 29): (21, 29)}

    occluded1 = ms.get_occluded_dets(dets1, close_iou_thrs=0, close_dist_thrs=0)
    occluded2 = ms.get_occluded_dets(dets2, close_iou_thrs=0, close_dist_thrs=0)
    matching_groups = ms.find_match_groups(dets1, dets2, occluded1, occluded2)
    n_occluded1, n_occluded2 = ms.get_not_occluded(dets1, dets2, matching_groups)

    assert occluded1 == exp_occluded1
    assert not n_occluded1.difference(exp_n_occluded) == True
    assert not n_occluded2.difference(exp_n_occluded) == True
    assert occluded2 == exp_occluded2
    assert matching_groups == exp_matching_groups


def test_get_cosim_matches_per_group():
    dummy_dets1 = np.array([[-1, 0, 13, 0, 0, 1, 1], [-1, 0, 44, 0, 0, 1, 1]])
    dummy_dets2 = np.array([[-1, 0, 13, 10, 10, 11, 11], [-1, 0, 44, 10, 10, 11, 11]])
    out = [44, 44, 83, 44, 13, 81, 13, 44, 82, 13, 13, 85]
    matches = ms.get_cosim_matches_per_group(out, dummy_dets1, dummy_dets2)
    exp_matched = [[13, 13], [44, 44]]
    assert exp_matched == matches


def test_stage1():
    occluded1 = [[6, 7], [13, 17], [21, 29]]
    occluded2 = [[13, 17], [21, 29]]
    flatten = [v for vv in occluded1 + occluded2 for v in vv]
    n_occluded = set(range(31)).difference(flatten)
    expected = list(zip(n_occluded, n_occluded))
    matched_dids = ms.get_n_occluded_matches(dets1, dets2, n_occluded, n_occluded)
    assert expected == matched_dids


def test_stage2():
    # TODO fix
    matching_groups = {(6, 7): (6, 7), (13, 17): (13, 17), (21, 29): (21, 29)}
    exp_occluded_matches = [
        (6, 6),  #  93
        (7, 7),  #  94
        (13, 13),  # 69
        (17, 17),  # 80
        (21, 21),  # 97
        (29, 29),  # 86
    ]
    occluded_matches = ms.get_occluded_matches(
        dets1, dets2, matching_groups, features1, features2
    )
    assert exp_occluded_matches == occluded_matches

    dets2[:, 2] = dets2[:, 0] + 10
    matching_groups = {(6, 7): (16, 17), (13, 17): (23, 27), (21, 29): (31, 39)}
    exp_occluded_matches = [
        (6, 16),  #  93
        (7, 17),  #  94
        (13, 23),  # 69
        (17, 27),  # 80
        (21, 31),  # 97
        (29, 39),  # 86
    ]
    features2_changed = {k + 10: v for k, v in features2.items()}
    occluded_matches = ms.get_occluded_matches(
        dets1, dets2, matching_groups, features1, features2_changed
    )
    assert exp_occluded_matches == occluded_matches


def test_handle_tracklets():
    exp_did2tid = {11: 1, 12: 2, 13: 3, 16: 6, 17: 7, 18: 8, 14: 4, 10: 9, 15: 10}
    exp_trks = np.array(
        [
            # [0, 184, 0, 1198, 448, 1217, 478, 1207, 463, 19, 30, 0, 0, 2],
            [1, 184, 1, 1127, 417, 1142, 445, 1135, 431, 15, 28, 100, 0, 1],
            [2, 184, 2, 1493, 452, 1510, 472, 1501, 462, 17, 20, 100, 0, 1],
            [3, 184, 3, 1075, 330, 1091, 340, 1083, 335, 16, 10, 100, 0, 1],
            [4, 184, 4, 1215, 301, 1227, 319, 1221, 310, 12, 18, 100, 0, 1],
            # [5, 184, 5, 1211, 317, 1223, 333, 1217, 325, 13, 15, 0, 0, 2],
            [6, 184, 6, 1076, 499, 1098, 513, 1087, 506, 22, 13, 100, 0, 1],
            [7, 184, 7, 1047, 480, 1074, 488, 1060, 484, 27, 8, 100, 0, 1],
            [8, 184, 8, 1171, 584, 1201, 610, 1186, 597, 30, 27, 100, 0, 1],
            [1, 192, 11, 1127, 417, 1141, 447, 1134, 432, 14, 29, 100, 0, 1],
            [2, 192, 12, 1493, 455, 1509, 475, 1501, 465, 17, 20, 100, 0, 1],
            [3, 192, 13, 1074, 331, 1090, 341, 1082, 336, 17, 10, 100, 0, 1],
            [6, 192, 16, 1076, 499, 1098, 513, 1087, 506, 22, 13, 100, 0, 1],
            [7, 192, 17, 1046, 481, 1073, 489, 1059, 485, 27, 8, 100, 0, 1],
            [8, 192, 18, 1171, 584, 1202, 611, 1187, 597, 31, 27, 100, 0, 1],
            [4, 192, 14, 1215, 301, 1227, 319, 1221, 310, 12, 18, 100, 0, 1],
            [9, 192, 10, 1197, 450, 1216, 481, 1206, 466, 19, 30, 100, 0, 1],
            [10, 192, 15, 1210, 318, 1223, 333, 1217, 326, 13, 15, 100, 0, 1],
        ]
    )

    vid_name, frame_number1, step, folder = 2, 184, 8, "240hz"
    main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
    frame_number2 = frame_number1 + step

    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

    dets1 = tracks[tracks[:, 1] == frame_number1]
    dets2 = tracks[tracks[:, 1] == frame_number2]
    # missuse tracks for detections
    dets1[:, 2] = dets1[:, 0]
    dets2[:, 2] = dets2[:, 0] + 10  # test if matches are correct for det_id
    dets1[:, 0] = -1
    dets2[:, 0] = -1
    extension = 100 * np.ones((len(dets2), 1), dtype=np.int64)
    dets2 = np.concatenate((dets2, extension), axis=1)

    features2_changed = {k + 10: v for k, v in features2.items()}
    matches = ms.get_matches(
        dets1, dets2, features1, features2_changed, close_dist_thrs=0
    )
    matches.remove((0, 10))
    matches.remove((5, 15))
    # [(1, 11), (2, 12), (3, 13), (6, 16), (7, 17), (8, 18), (4, 14)]
    extension = np.repeat(np.array([[100, 0, 1]]), len(dets1), axis=0)
    trks = dets1.copy()
    trks[:, 0] = trks[:, 2]
    trks = np.concatenate((trks, extension), axis=1)
    dets1 = ms.get_last_dets_tracklets(trks)
    trks, did2tid, _ = ms.handle_tracklets(dets1, dets2, matches, trks)

    np.testing.assert_array_equal(trks, exp_trks)
    assert did2tid == exp_did2tid


def test_kill_tracks():
    vid_name, folder = 2, "240hz"
    main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")

    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
    extension = np.zeros((len(tracks), 3), dtype=np.int64)
    tracks = np.concatenate((tracks, extension), axis=1)

    _ = ms.kill_tracks(tracks, 3117, 50)
    assert tracks[30193, 13] == 0
    _ = ms.kill_tracks(tracks, 4000, 50)
    assert tracks[30193, 13] == 3


def test_improve_hungarian():
    cost_matrix = np.ones(shape=(4, 3)).astype(np.float32)
    cost_matrix[0, 1] = 0.2
    cost_matrix[2, 2] = 0.3
    cost_matrix[3, 2] = 0.4

    ass_rows, ass_cols = ms.improve_hungarian(cost_matrix, thrs=0.8)
    assert ass_rows == [0, 2]
    assert ass_cols == [1, 2]


def test_remove_values_from_dict():
    original_dic = {(6, 7): (5, 3), (15, 17, 18): (2, 7, 1)}
    remove_values = [3, 1, 2]
    expected = {(6, 7): (5,), (15, 17, 18): (7,)}
    final_dic = ms.remove_values_from_dict(original_dic, remove_values)
    assert expected == final_dic


test_improve_hungarian()
test_merge_intersecting_lists()
test_get_occluded_dets()
test_merge_overlapping_keys()
test_merge_overlapping_keys_and_values()
test_find_match_groups()
test_get_cosim_matches_per_group()
test_stage1()
test_stage2()
test_handle_tracklets()
test_kill_tracks()
test_remove_values_from_dict()
print("passed")
