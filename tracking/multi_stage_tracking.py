# - group occluded
# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from itertools import combinations
from pathlib import Path

from tracking import data_association as da

main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")

vid_name = 6  # 2
frame_number1 = 16  # 192
frame_number2 = 24  # 184

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]

# import cv2
# import matplotlib.pylab as plt
# from tracking import visualize
# image1 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
# image2 = cv2.imread(str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
# visualize.plot_detections_in_image(dets1[:,[0,3,4,5,6]], image1);plt.show(block=False)
# visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], image2);plt.show(block=False)


def get_occluded_dets(dets):
    occluded = {}
    n_occluded = []
    ids = dets[:, 0]  # TODO change to det id
    for tid1, tid2 in combinations(ids, 2):
        det1 = dets[dets[:, 0] == tid1][0]  # TODO
        det2 = dets[dets[:, 0] == tid2][0]  # TODO
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(tid1, [tid1]).append(tid2)
    occluded = list(occluded.values())
    flatten = [v for vv in occluded for v in vv]
    n_occluded = set(ids).difference(flatten)
    return occluded, n_occluded


def find_match_groups(occluded1, occluded2):
    matching_groups = {}
    for group1 in occluded1:
        group1 = tuple(sorted(set(group1)))
        values = []
        for tid1 in group1:
            det1 = dets1[dets1[:, 0] == tid1][0]  # TODO
            for det2 in dets2:
                tid2 = det2[0]  # TODO
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(tid2)
        group2 = tuple(sorted(set(values)))
        matching_groups[group1] = group2
    for group2 in occluded2:
        values = []
        for tid2 in group2:
            det2 = dets2[dets2[:, 0] == tid2][0]  # TODO
            for det1 in dets1:
                tid1 = det1[0]  # TODO
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(tid1)
        group1 = tuple(sorted(set(group1)))
        if group1 not in matching_groups.keys():
            matching_groups[group1] = group2
    return matching_groups


occluded1, n_occluded1 = get_occluded_dets(dets1)
occluded2, n_occluded2 = get_occluded_dets(dets2)
matching_groups = find_match_groups(occluded1, occluded2)

print(occluded1, n_occluded1)
print(occluded2, n_occluded2)
print(matching_groups)


# =============================
main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")
vid_name = 6 
frame_number1 = 16 
frame_number2 = 24 
tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]

def test_find_match_groups():
    exp_occluded1 = [[6, 7], [13, 17], [21, 29]]
    flatten = [v for vv in exp_occluded1 for v in vv]
    exp_n_occluded1 = set(range(31)).difference(flatten)
    # {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31}
    exp_occluded2 = [[13, 17], [21, 29]]
    exp_matching_groups = {(6, 7): (6, 7), (13, 17): (13, 17), (21, 29): (21, 29)}

    occluded1, n_occluded1 = get_occluded_dets(dets1)
    occluded2, _ = get_occluded_dets(dets2)
    matching_groups = find_match_groups(occluded1, occluded2)

    assert occluded1 == exp_occluded1
    assert not n_occluded1.difference(exp_n_occluded1) == True
    assert occluded2 == exp_occluded2
    assert matching_groups == exp_matching_groups


test_find_match_groups()
