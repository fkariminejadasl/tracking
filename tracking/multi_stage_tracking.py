# - group occluded
# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from pathlib import Path
import cv2
from tracking import data_association as da
import matplotlib.pylab as plt
from tracking import visualize
from itertools import combinations, product

main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")

vid_name = 2  # 6
frame_number1 = 192  # 0
frame_number2 = 184  # 8

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

image1 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
image2 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]

# visualize.plot_detections_in_image(dets1[:,[0,3,4,5,6]], image1);plt.show(block=False)
# visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], image2);plt.show(block=False)


def get_occluded_dets(dets):
    # dets = dets1.copy()
    occluded = {}
    n_occluded = []
    ids = dets[:, 0]  # TODO change to det id
    for tid1, tid2 in combinations(ids, 2):
        det1 = dets[dets[:, 0] == tid1][0]
        det2 = dets[dets[:, 0] == tid2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(tid1, [tid1]).append(tid2)
    occluded = list(occluded.values())
    flatten = [v for vv in occluded for v in vv]
    n_occluded = set(ids).difference(flatten)
    return occluded, n_occluded


def get_occluded_dets_two_frames(dets1, dets2):
    occluded = {}
    ids1 = dets1[:, 0]  # TODO change to det id
    ids2 = dets2[:, 0]  # TODO change to det id
    for tid1, tid2 in product(ids1, ids2):
        det1 = dets1[dets1[:, 0] == tid1][0]
        det2 = dets2[dets2[:, 0] == tid2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(tid1, []).append(tid2)
    return occluded


def make_a_occluded_groups(occluded, if_occluded):
    g_occluded = {}
    for k, vv in occluded.items():
        g_occluded[k] = []
        for v in vv:
            g_occluded[k] += if_occluded[v]
        g_occluded[k] = list(set(g_occluded[k]))
    return g_occluded


occluded1, n_occluded1 = get_occluded_dets(dets1)
occluded2, n_occluded2 = get_occluded_dets(dets2)
# if_occluded1 = get_occluded_dets_two_frames(dets1, dets2)  # interframe occlusion
# if_occluded2 = get_occluded_dets_two_frames(dets1, dets2)  # interframe occlusion
# g_occluded1 = make_a_occluded_groups(occluded1, if_occluded1)
# g_occluded2 = make_a_occluded_groups(occluded2, if_occluded2)

# def find_match_groups(occluded, if_occluded):
matching_groups = {}
for group1 in occluded1:
    values = []
    for tid1 in group1:
        det1 = dets1[dets1[:, 0] == tid1][0]  # TODO
        for det2 in dets2:
            tid2 = det2[0]  # TODO
            if da.get_iou(det1[3:7], det2[3:7]) > 0:
                values.append(tid2)
    group2 = list(set(values))
    matching_groups[tuple(group1)] = group2
for group2 in occluded2:
    values = []
    for tid2 in group2:
        det2 = dets2[dets2[:, 0] == tid2][0]  # TODO
        for det1 in dets1:
            tid1 = det1[0]  # TODO
            if da.get_iou(det1[3:7], det2[3:7]) > 0:
                values.append(tid1)
    group1 = tuple(set(values))
    if group1 not in matching_groups.keys():
        matching_groups[group1] = group2

# matching_groups = {}
# for i in range(len(occluded1)):
#     matching_groups[tuple(occluded1[i])] = occluded2[i]
print(matching_groups)

print(occluded1, n_occluded1)
print(occluded2, n_occluded2)
# print(if_occluded1)
# print(if_occluded2)
# print(g_occluded1)
# print(g_occluded2)

# =============================
main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")
vid_name = 2  # 6
frame_number1 = 0
frame_number2 = 8


def test_get_occluded_dets():
    occluded, n_occluded = get_occluded_dets(dets1)
    exp_occluded = {4: [4, 5]}
    exp_n_occluded = {0, 1, 2, 3, 6, 7, 8}
    assert n_occluded == exp_n_occluded
    assert occluded == exp_occluded


def test_get_occluded_dets_two_frames():
    occluded = get_occluded_dets_two_frames(dets1, dets2)
    exp_occluded = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4, 5],
        5: [4, 5],
        6: [6],
        7: [7],
        8: [8],
    }
    assert occluded == exp_occluded


# test_get_occluded_dets()
# test_get_occluded_dets_two_frames()
