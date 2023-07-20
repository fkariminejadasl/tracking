# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from itertools import combinations
from pathlib import Path

import numpy as np

from tracking import data_association as da

main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")

vid_name = 6  # 2
frame_number1 = 16  # 192
frame_number2 = 24  # 184

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]
# TODO hack to missuse tracks for detections
dets1[:, 2] = dets1[:, 0]
dets2[:, 2] = dets2[:, 0]

# import cv2
# import matplotlib.pylab as plt
# from tracking import visualize
# image1 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
# image2 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
# visualize.plot_detections_in_image(dets1[:,[0,3,4,5,6]], image1);plt.show(block=False)
# visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], image2);plt.show(block=False)


def get_occluded_dets(dets):
    occluded = {}
    ids = dets[:, 2]
    for did1, did2 in combinations(ids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(did1, [did1]).append(did2)
    occluded = list(occluded.values())
    return occluded


def find_match_groups(occluded1, occluded2):
    matching_groups = {}
    for group1 in occluded1:
        group1 = tuple(sorted(set(group1)))
        values = []
        for did1 in group1:
            det1 = dets1[dets1[:, 2] == did1][0]
            for det2 in dets2:
                did2 = det2[2]
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(did2)
        group2 = tuple(sorted(set(values)))
        matching_groups[group1] = group2
    for group2 in occluded2:
        values = []
        for did2 in group2:
            det2 = dets2[dets2[:, 2] == did2][0]
            for det1 in dets1:
                did1 = det1[2]
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(did1)
        group1 = tuple(sorted(set(group1)))
        if group1 not in matching_groups.keys():
            matching_groups[group1] = group2
    return matching_groups


def get_not_occluded(dets1, dets2, matching_groups):
    dids1 = dets1[:, 2]
    group = matching_groups.keys()
    flatten = [v for vv in group for v in vv]
    n_occluded1 = set(dids1).difference(flatten)
    dids2 = dets2[:, 2]
    group = matching_groups.values()
    flatten = [v for vv in group for v in vv]
    n_occluded2 = set(dids2).difference(flatten)
    return n_occluded1, n_occluded2


occluded1 = get_occluded_dets(dets1)
occluded2 = get_occluded_dets(dets2)
matching_groups = find_match_groups(occluded1, occluded2)
n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

print(occluded1, n_occluded1)
print(occluded2, n_occluded2)
print(matching_groups)

# Stage 1: Hungarian matching on non occluded detections
s_dets1 = np.array([dets2[dets2[:, 2] == did][0] for did in n_occluded1])
s_dets2 = np.array([dets2[dets2[:, 2] == did][0] for did in n_occluded2])
sc_dets1 = da.make_dets_from_array(s_dets1)
sc_dets2 = da.make_dets_from_array(s_dets2)
inds1, inds2 = da.hungarian_global_matching(sc_dets1, sc_dets2)
matched_dids = [
    (s_dets1[ind1, 2], s_dets2[ind2, 2]) for ind1, ind2 in zip(inds1, inds2)
]
print(matched_dids)

# Stage 2: Cos similarity of concatenated embeddings


# =============================
main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")
vid_name = 6
frame_number1 = 16
frame_number2 = 24
tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]
# TODO hack to missuse tracks for detections
dets1[:, 2] = dets1[:, 0]
dets2[:, 2] = dets2[:, 0]


def test_find_match_groups():
    exp_occluded1 = [[6, 7], [13, 17], [21, 29]]
    exp_occluded2 = [[13, 17], [21, 29]]
    flatten = [v for vv in exp_occluded1 + exp_occluded2 for v in vv]
    exp_n_occluded = set(range(31)).difference(flatten)
    # {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31}
    exp_matching_groups = {(6, 7): (6, 7), (13, 17): (13, 17), (21, 29): (21, 29)}

    occluded1 = get_occluded_dets(dets1)
    occluded2 = get_occluded_dets(dets2)
    matching_groups = find_match_groups(occluded1, occluded2)
    n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

    assert occluded1 == exp_occluded1
    assert not n_occluded1.difference(exp_n_occluded) == True
    assert not n_occluded2.difference(exp_n_occluded) == True
    assert occluded2 == exp_occluded2
    assert matching_groups == exp_matching_groups


test_find_match_groups()
