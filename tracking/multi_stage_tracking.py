# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torchvision
from scipy.optimize import linear_sum_assignment

from tracking import data_association as da

np.random.seed(1000)


def get_occluded_dets(dets):
    """
    input:
        dets np.ndarray
    output: list(list(int))
    output e.g. [[13, 17], [21, 29]]
    """

    # TODO if 8, 9, 11, where 9, 11 intersect, 8, 11 but not 8, 9. This return two groups.
    # if 11 was a smaller number then one group of three is return. I'm not sure if I change
    # this part.
    occluded = {}
    ids = dets[:, 2]
    for did1, did2 in combinations(ids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(did1, [did1]).append(did2)
    occluded = list(occluded.values())
    return occluded


def find_match_groups(dets1, dets2, occluded1, occluded2):
    """
    inputs:
        dets1, dets2: np.ndarray
        occluded1, occluded2: list(list(int))
    output: dic(tuple(int), tuple(int))

    output e.g. {(6, 7): (6, 7), (13, 17): (13, 17)
    """
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
    """
    inputs:
        dets1, dets2: np.ndarray
        matching_groups: dic(tuple(int), tuple(int))
    output: tuple(set(int), set(int))
    """
    dids1 = dets1[:, 2]
    group = matching_groups.keys()
    flatten = [v for vv in group for v in vv]
    n_occluded1 = set(dids1).difference(flatten)
    dids2 = dets2[:, 2]
    group = matching_groups.values()
    flatten = [v for vv in group for v in vv]
    n_occluded2 = set(dids2).difference(flatten)
    return n_occluded1, n_occluded2


def get_model_args():
    device = "cuda"
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    ).to(device)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    model.eval()
    model.requires_grad_(False)

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    activation = {}
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.layer1.register_forward_hook(get_activation("layer1"))
    model.layer2.register_forward_hook(get_activation("layer2"))
    model.layer3.register_forward_hook(get_activation("layer3"))
    model.layer4.register_forward_hook(get_activation("layer4"))

    kwargs = {
        "model": model,
        "device": device,
        "transform": transform,
        "activation": activation,
    }
    return kwargs


def hungarian_global_matching(dets1, dets2):
    """
    inputs:
        dets1, dets2: np.ndarray
    output: tuple(list(int), list(int))
    """
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - da.get_iou(det1[3:7], det2[3:7])
            loc_loss = np.linalg.norm([det2[7] - det1[7], det2[8] - det1[8]])
            dist[i, j] = iou_loss + loc_loss
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2):
    """
    inputs:
        dets1, dets2: np.ndarray
        n_occluded1, n_occluded2: set(int)
    output: list[tuple(int, int)]
    """
    s_dets1 = np.array([dets1[dets1[:, 2] == did][0] for did in n_occluded1])
    s_dets2 = np.array([dets2[dets2[:, 2] == did][0] for did in n_occluded2])
    inds1, inds2 = hungarian_global_matching(s_dets1, s_dets2)
    matched_dids = [
        (s_dets1[ind1, 2], s_dets2[ind2, 2]) for ind1, ind2 in zip(inds1, inds2)
    ]

    return matched_dids


def bbox_enlarge(bbox, w_enlarge=0, h_enlarge=0):
    n_bbox = deepcopy(bbox)
    n_bbox[3] -= w_enlarge
    n_bbox[5] += w_enlarge
    n_bbox[4] -= h_enlarge
    n_bbox[6] += h_enlarge
    n_bbox[9] = n_bbox[5] - n_bbox[3]
    n_bbox[10] = n_bbox[6] - n_bbox[4]
    return n_bbox


def clip_bboxs(bbox, im_height, im_width):
    bbox[:, 3:6:2] = np.clip(bbox[:, 3:6:2], 0, im_width - 1)
    bbox[:, 4:7:2] = np.clip(bbox[:, 4:7:2], 0, im_height - 1)
    return bbox


def cos_sim(main_path, vid_name, bbs1, bbs2, **kwargs):
    """
    inputs:
        main_path: Path
        vid_name: str|int
        bbs1, bbs2: np.ndarray
    output: list[int]
    e.g. output [44, 44, 83, 44, 13, 81, 13, 44, 82, 13, 13, 85]
    """
    model = kwargs.get("model")
    transform = kwargs.get("transform")
    device = kwargs.get("device")
    activation = kwargs.get("activation")

    bbs1 = np.array([bbox_enlarge(bb) for bb in bbs1])
    bbs2 = np.array([bbox_enlarge(bb) for bb in bbs2])
    w, h = max(max(bbs1[:, -2]), max(bbs2[:, -2])), max(
        max(bbs1[:, -1]), max(bbs2[:, -1])
    )

    frame_number2 = bbs2[0, 1]
    im2 = cv2.imread(
        str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg")
    )
    im_height, im_width, _ = im2.shape
    clip_bboxs(bbs1, im_height, im_width)
    clip_bboxs(bbs2, im_height, im_width)

    # print("concate embeddings")
    layers = ["conv1", "layer1", "layer2", "layer3"]
    output = []
    for bb1 in bbs1:
        for bb2 in bbs2:
            frame_number1 = bb1[1]
            im1 = cv2.imread(
                str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg")
            )

            imc1 = im1[bb1[4] : bb1[6], bb1[3] : bb1[5]]
            imc2 = im2[bb2[4] : bb2[6], bb2[3] : bb2[5]]
            imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
            imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
            _ = model(transform(imc1).unsqueeze(0).to(device))
            f1 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            _ = model(transform(imc2).unsqueeze(0).to(device))
            f2 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            csim = int(np.round(csim * 100))  # percent
            output.extend([bb1[0], bb2[0], csim])
    return output


def get_cosim_matches_per_group(out):
    """
    input:
        out: list(int)
    output: list(list(int))
    input e.g. [44, 44, 83, 44, 13, 81, 13, 44, 82, 13, 13, 85]
    output e.g. [[13, 13, 85], [44, 44, 83]]
    """
    out = np.array(out).reshape(-1, 3)
    ids1 = np.unique(out[:, 0])
    ids2 = np.unique(out[:, 1])
    matches = []
    dist = np.zeros((len(ids1), len(ids2)), dtype=np.float32)
    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            cosim = out[(out[:, 0] == id1) & (out[:, 1] == id2)][:, 2]
            dist[i, j] = 1 - cosim / 100
    row_inds, col_inds = linear_sum_assignment(dist)
    for row_ind, col_ind in zip(row_inds, col_inds):
        cosim = int(round((1 - dist[row_ind, col_ind]) * 100))
        matches.append([ids1[row_ind], ids2[col_ind], cosim])
    # TODO something here to distiguish low quality ass and multi dets
    # tricky for multiple dets, low quality. I cover mis det in unmatched
    return matches


def get_occluded_matches_per_group(main_path, vid_name, bbs1, bbs2, **kwargs):
    """
    inputs:
        main_path: Path
        vid_name: str|int
        bbs1, bbs2: np.ndarray
    output: list(tuple(int, int))
    """
    out = cos_sim(main_path, vid_name, bbs1, bbs2, **kwargs)
    matches = get_cosim_matches_per_group(out)
    return matches


def get_bboxes(dets: np.ndarray, group):
    bbs = []
    for id_ in group:
        bbs.append(dets[(dets[:, 2] == id_)])
    bbs = np.concatenate(bbs, axis=0)
    return bbs


def get_occluded_matches(dets1, dets2, matching_groups, main_path, vid_name, **kwargs):
    """
    inputs:
        dets1, dets2: np.ndarray
        matching_groups: dict(tuple, tuple)
        main_path: Path
        vid_name: str|int
    output: list(tuple(int, int))
    """
    occluded_matches = []
    for group1, group2 in matching_groups.items():
        bbs1 = get_bboxes(dets1, group1)
        bbs2 = get_bboxes(dets2, group2)
        cosim_matches_group = get_occluded_matches_per_group(
            main_path, vid_name, bbs1, bbs2, **kwargs
        )
        occluded_matches.extend(
            [tuple(cosim_match_group[:2]) for cosim_match_group in cosim_matches_group]
        )
    return occluded_matches


def get_matches(dets1, dets2, main_path, vid_name, **kwargs):
    """
    inputs:
        dets1, dets2: np.ndarray
        dets1: can be the (predicted) last detections of tracklets or the image detections
        dets2: is the image detections
    """
    occluded1 = get_occluded_dets(dets1)
    occluded2 = get_occluded_dets(dets2)
    matching_groups = find_match_groups(dets1, dets2, occluded1, occluded2)
    n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

    # Stage 1: Hungarian matching on non occluded detections
    n_occluded_matches = get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2)

    # Stage 2: Cos similarity of concatenated embeddings
    occluded_matches = get_occluded_matches(
        dets1, dets2, matching_groups, main_path, vid_name, **kwargs
    )

    print(occluded1, n_occluded1)
    print(occluded2, n_occluded2)
    print(matching_groups)
    print(n_occluded_matches)
    print(occluded_matches)

    return n_occluded_matches + occluded_matches


def handle_unmatched(dets1, dets2, unmatched1, unmatched2):
    pass


def kill_tracks():
    pass


kwargs = get_model_args()  # TODO ugly
# TODO
# 1. s1: hungarian dist&iou on high quality dets no overlap (I have no_ovelap version)
# 2. s2: hungarian agg cossim on coverlap -> low quality ass (either low value or multiple detection)
# 3. s3: hungarian dist&iou on low quality dets
# 4. unmached (tracklets: mis track but keept, dets: new track)
# 5. kill track (not tracked for long time)

"""
# stay for a while for some simple tests

# 6, 16, 8, "240hz"  # 2, 184, 8, "240hz", # 0, 38, 1, "30hz"
vid_name, frame_number1, step, folder = 2, 184, 8, "240hz"
main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
frame_number2 = frame_number1 + step

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
# dets1[8] = tracks[(tracks[:, 1] == 36) & (tracks[:,0]==8)][0] # test for tracklet and frame
dets2 = tracks[tracks[:, 1] == frame_number2]
# TODO hack to missuse tracks for detections
dets1[:, 2] = dets1[:, 0]
dets2[:, 2] = dets2[:, 0]


# import matplotlib.pylab as plt
# from tracking import visualize
# visualize.plot_detections_in_image(dets1[:,[0,3,4,5,6]], im1);plt.show(block=False)
# visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], im2);plt.show(block=False)

occluded1 = get_occluded_dets(dets1)
occluded2 = get_occluded_dets(dets2)
matching_groups = find_match_groups(dets1, dets2, occluded1, occluded2)
n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

print(occluded1, n_occluded1)
print(occluded2, n_occluded2)
print(matching_groups)


# Stage 1: Hungarian matching on non occluded detections
n_occluded_matches = get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2)
print(n_occluded_matches)

# Stage 2: Cos similarity of concatenated embeddings
occluded_matches = get_occluded_matches(
    dets1, dets2, matching_groups, main_path, vid_name, **kwargs
)
print(occluded_matches)

# matches = get_matches(dets1, dets2, main_path, vid_name, **kwargs)
# print("all together:", matches)
"""

# =============================
kwargs = get_model_args()
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
    matching_groups = find_match_groups(dets1, dets2, occluded1, occluded2)
    n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

    assert occluded1 == exp_occluded1
    assert not n_occluded1.difference(exp_n_occluded) == True
    assert not n_occluded2.difference(exp_n_occluded) == True
    assert occluded2 == exp_occluded2
    assert matching_groups == exp_matching_groups


def test_get_cosim_matches_per_group():
    out = [44, 44, 83, 44, 13, 81, 13, 44, 82, 13, 13, 85]
    matches = get_cosim_matches_per_group(out)
    exp_matched = [[13, 13, 85], [44, 44, 83]]
    assert exp_matched == matches


def test_stage1():
    occluded1 = [[6, 7], [13, 17], [21, 29]]
    occluded2 = [[13, 17], [21, 29]]
    flatten = [v for vv in occluded1 + occluded2 for v in vv]
    n_occluded = set(range(31)).difference(flatten)
    expected = list(zip(n_occluded, n_occluded))
    matched_dids = get_n_occluded_matches(dets1, dets2, n_occluded, n_occluded)
    assert expected == matched_dids


def test_stage2():
    matching_groups = {(6, 7): (6, 7), (13, 17): (13, 17), (21, 29): (21, 29)}
    exp_occluded_matches = [
        (6, 6),  #  93
        (7, 7),  #  94
        (13, 13),  # 69
        (17, 17),  # 80
        (21, 21),  # 97
        (29, 29),  # 86
    ]
    occluded_matches = get_occluded_matches(
        dets1, dets2, matching_groups, main_path, vid_name, **kwargs
    )
    assert exp_occluded_matches == occluded_matches


test_find_match_groups()
test_get_cosim_matches_per_group()
test_stage1()
test_stage2()
print("passed")
