# - use Hungarian for each not occluded
# - use Cosim for each occluded group
from collections import Counter
from copy import deepcopy
from itertools import chain, combinations

import cv2
import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO

from tracking import data_association as da

np.random.seed(1000)

# TODO put it in get_model_args
layers = ["conv1", "layer1", "layer2", "layer3"]
disp_thrs = 15
thrs_iou = 0.3  # .28
thrs_inside = 3  # pixel
# improve Hungarian threshold
hug_thrs = 0.8
# for close bboxes
close_iou_thrs = 0
close_dist_thrs = 3
"""
In ultralytics botsort, match_thresh=.8 is used for hungarian but it is after matching.
My improve_hungarian handles it better by totally removing the high cost vlaues. This 
is done by first thresholding cost function and then removing all zero row and columns.  
TODO:
- det_score x iou in fuse_score in 
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/matching.py
- new_track_thresh=.6
"""


def large_occlusion(det1, det2, thrs_iou, thrs_inside):
    """
    Identify large occlusions
    """
    is_inside2 = da.is_inside_bbox(det1[3:7], det2[3:7], thrs_inside)
    is_inside1 = da.is_inside_bbox(det2[3:7], det1[3:7], thrs_inside)
    overlap = da.get_iou(det1[3:7], det2[3:7]) > thrs_iou
    return is_inside1 or is_inside2 or overlap


def identify_nms_dids(dets, thrs_iou, thrs_inside):
    """
    Identify NMS (non maximum suppression) detection ids
    """
    dids = dets[:, 2]
    rm_dids = []
    for did1, did2 in combinations(dids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        is_occluded = large_occlusion(det1, det2, thrs_iou, thrs_inside)
        if is_occluded:
            if det1[11] < det2[11]:  # dq (det quality)
                rm_dids.append(det1[2])
            else:
                rm_dids.append(det2[2])
    return list(set(rm_dids))


def non_max_sup(dets, thrs_iou, thrs_inside):
    """
    Identify large occluded detections and remove the one with lowest detection quality (dq)
    """
    r_dids = identify_nms_dids(dets, thrs_iou, thrs_inside)
    rows_to_keep = ~np.isin(dets[:, 2], r_dids)
    dets = dets[rows_to_keep]
    return dets


def merge_intersecting_lists(lst):
    """
    input: list[list[int]]
    output: list[list[int]]

    e.g. merge_intersecting_lists([[1,2], [3,2], [3,4], [5,6], [7], [6,8]]) returns
    [[1, 2, 3, 4], [5, 6, 8], [7]]
    """

    def merge_sort(sublist):
        sublist.sort()
        return sublist

    result = []
    for sublist in lst:
        merged = False
        for existing in result:
            if any(item in existing for item in sublist):
                existing.extend(item for item in sublist if item not in existing)
                existing = merge_sort(existing)  # Sort the merged sublist
                merged = True
                break
        if not merged:
            result.append(merge_sort(sublist))  # Sort and add the new sublist
    return result


def get_occluded_dets2(dets):
    """
    Get only occluded

    input:
        dets np.ndarray
    output: list[list[int]]
        The values are the detection ids.

    output e.g. [[10, 15, 17], [21, 29]]
    """

    # if 8, 9, 11, where 9 and 11 intersect and 8 and 11 intersect but not 8, 9. This return two groups.
    # if 11 was a smaller number then one group of three is return. I'm not sure if I change
    # this part. -> now is changed by merge_intersecting_lists
    occluded = {}
    ids = dets[:, 2]
    for did1, did2 in combinations(ids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(did1, [did1]).append(did2)
    occluded = list(occluded.values())
    return merge_intersecting_lists(occluded)


def get_occluded_dets(
    dets, close_iou_thrs=close_iou_thrs, close_dist_thrs=close_dist_thrs
):
    """
    Get occluded and nearly occluded.
    If close_iou_thrs=0 and close_dist_thrs=0, it is the same as get_occluded_dets2

    input:
        dets np.ndarray
    output: list[list[int]]
        The values are the detection ids.

    output e.g. [[10, 15, 17], [21, 29]]
    """

    # if 8, 9, 11, where 9 and 11 intersect and 8 and 11 intersect but not 8, 9. This return two groups.
    # if 11 was a smaller number then one group of three is return. I'm not sure if I change
    # this part. -> now is changed by merge_intersecting_lists
    occluded = {}
    ids = dets[:, 2]
    for did1, did2 in combinations(ids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        is_close = da.are_boxes_close(
            det1[3:7], det2[3:7], iou_thrs=close_iou_thrs, dist_thrs=close_dist_thrs
        )
        if is_close:
            occluded.setdefault(did1, [did1]).append(did2)
    occluded = list(occluded.values())
    return merge_intersecting_lists(occluded)


def merge_overlapping_keys(input):
    """
    input, output: dic[tuple[int, ...], tuple[int, ...]]

    input output e.g. {(6, 7): (1,), (6,): (2, 3), (7, 8): (5,)} -> {(6, 7, 8): (1, 2, 3, 5)}
    """
    merged = {}
    while input:
        key1, value1 = input.popitem()
        rest = deepcopy(input)
        is_merged = False
        for key2, value2 in rest.items():
            if set(key2).intersection(key1):
                merged_key = tuple(sorted(set(key1 + key2)))
                merged_value = tuple(sorted(set(value1 + value2)))
                merged[merged_key] = merged_value
                key1 = merged_key
                value1 = merged_value
                _ = input.pop(key2)
                is_merged = True
        if not is_merged:
            merged[key1] = value1
    return merged


def merge_overlapping_keys_and_values(input):
    """
    input, output: dic[tuple[int, ...], tuple[int, ...]]

    input output e.g. {(6,7):(1,), (6,):(2,3), (4,):(4,), (9,8):(5,1)} -> {(6, 7, 8, 9): (1, 2, 3, 5), (4,):(4,)}
    """

    # merge keys
    output = merge_overlapping_keys(input)
    output = {val: key for key, val in output.items()}
    # merge values
    output = merge_overlapping_keys(output)
    output = {val: key for key, val in output.items()}
    return output


def find_match_groups(dets1, dets2, occluded1, occluded2):
    # TODO efficient implementation
    """
    inputs:
        dets1, dets2: np.ndarray
        occluded1, occluded2: list[list[int]]
    output: dic[tuple[int, ...], tuple[int, ...]]
        The values are the detection ids.

    output e.g. {(6, 7): (6, 7), (15, 17): (15, 17)}
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

    # Solution to the following problem:
    # in_sample_vids/240hz/vids/2.mp4, pre_frame = 1952, cur_frame = 1960, matches: 4<->4, 0<->1
    # occluded1=[4,17], occluded2=[], matching group (4,17):(3,4) because 4 in pre_frame intersects 3,4 in cur_frrame.
    # So 0 doesn't appear in occluded part and goes to stage1.
    occluded2 = merge_intersecting_lists(
        [list(val) for val in matching_groups.values()] + occluded2
    )

    for group2 in occluded2:
        group2 = tuple(sorted(set(group2)))
        values = []
        for did2 in group2:
            det2 = dets2[dets2[:, 2] == did2][0]
            for det1 in dets1:
                did1 = det1[2]
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(did1)
        group1 = tuple(sorted(set(values)))
        matching_groups[group1] = group2
    matching_groups = merge_overlapping_keys_and_values(matching_groups)
    return matching_groups


def get_not_occluded(dets1, dets2, matching_groups):
    """
    inputs:
        dets1, dets2: np.ndarray
        matching_groups: dic[tuple[int, ...], tuple[int, ...]]
    output: tuple[set[int], set[int]]
        The values are the detection ids.
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def improve_hungarian(cost, thrs):
    """
    inputs:
        cost: np.ndarray
            cost or loss matrix
    return:
        associated rows and columns: tuple[list[int], list[int]]
            The values are the indices of original cost matrix.
    """
    mask = cost < thrs
    rm_cols = np.where(np.all(mask == False, axis=0))[0]
    rm_rows = np.where(np.all(mask == False, axis=1))[0]
    old_rows = set(np.arange(cost.shape[0])).difference(rm_rows)
    old_cols = set(np.arange(cost.shape[1])).difference(rm_cols)
    new2old_row = {i: item for i, item in enumerate(old_rows)}
    new2old_col = {i: item for i, item in enumerate(old_cols)}
    new_cost = np.delete(np.delete(cost, rm_rows, axis=0), rm_cols, axis=1)
    rows, cols = linear_sum_assignment(new_cost)
    # TODO maybe add this one
    # row_cols = np.array([(row,col) for row, col in zip(rows, cols) if cost[row,col]<thrs])
    # rows = list(row_cols[:, 0])
    # cols = list(row_cols[:, 1])
    ass_rows = [new2old_row[item] for item in rows]
    ass_cols = [new2old_col[item] for item in cols]
    return ass_rows, ass_cols


def hungarian_global_matching(dets1, dets2):
    """
    inputs:
        dets1, dets2: np.ndarray
    output: tuple[list[int], list[int]]
        The values are the indices.
    """
    dist = np.zeros((len(dets1), len(dets2)), dtype=np.float32)
    for i, det1 in enumerate(dets1):
        for j, det2 in enumerate(dets2):
            iou_loss = 1 - da.get_iou(det1[3:7], det2[3:7])
            # loc_loss = np.linalg.norm([det2[7] - det1[7], det2[8] - det1[8]])
            dist[i, j] = iou_loss  # + loc_loss
    # row_ind, col_ind = linear_sum_assignment(dist)
    row_ind, col_ind = improve_hungarian(dist, thrs=hug_thrs)
    return row_ind, col_ind


def get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2):
    """
    inputs:
        dets1, dets2: np.ndarray
        n_occluded1, n_occluded2: set[int]
            set of the detection ids
    output: list[tuple[int, int]]
        The values are the detection ids.
    """
    s_dets1 = np.array([dets1[dets1[:, 2] == did][0] for did in n_occluded1])
    s_dets2 = np.array([dets2[dets2[:, 2] == did][0] for did in n_occluded2])
    inds1, inds2 = hungarian_global_matching(s_dets1, s_dets2)
    matched_dids = [
        (s_dets1[ind1, 2], s_dets2[ind2, 2]) for ind1, ind2 in zip(inds1, inds2)
    ]

    return matched_dids


def clip_bboxs(bbox, im_height, im_width):
    bbox[:, 3:6:2] = np.clip(bbox[:, 3:6:2], 0, im_width - 1)
    bbox[:, 4:7:2] = np.clip(bbox[:, 4:7:2], 0, im_height - 1)
    return bbox


def cos_sim(features1, features2, bbs1, bbs2):
    """
    inputs:
        image_path: Path
        vid_name: str | int
        bbs1, bbs2: np.ndarray
    output: list[int]
        The values are the detection ids.

    e.g. output [44, 44, 83, 44, 15, 81, 15, 44, 82, 15, 15, 85]
    """
    output = []
    for bb1 in bbs1:
        for bb2 in bbs2:
            f1 = features1[bb1[2]].copy()
            f2 = features2[bb2[2]].copy()
            csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            csim = int(np.round(csim * 100))  # percent
            output.extend([bb1[2], bb2[2], csim])
    return output


def calculate_deep_feature(image, layers, w=32, h=32, **kwargs):
    model = kwargs.get("model")
    transform = kwargs.get("transform")
    device = kwargs.get("device")
    activation = kwargs.get("activation")

    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    _ = model(transform(image).unsqueeze(0).to(device))
    feat = np.concatenate(
        [activation[layer].flatten().cpu().numpy() for layer in layers]
    )
    return feat


def calculate_deep_features(det_ids, dets, image, **kwargs):
    features = dict()
    if len(det_ids) != 0:
        bbs1 = get_bboxes(dets, det_ids)
        for bb1 in bbs1:
            imc = image[bb1[4] : bb1[6], bb1[3] : bb1[5]]
            did = bb1[2]
            features[did] = calculate_deep_feature(imc, layers, w=32, h=32, **kwargs)
    return features


def get_cosim_matches_per_group(out, dets1, dets2):
    """
    input:
        out: list[int]
    output: list[list[int]]
        The values are the detection ids and cosine similarity.

    input e.g. [44, 44, 83, 44, 15, 81, 15, 44, 82, 15, 15, 85]
    output e.g. [[15, 15, 85], [44, 44, 83]]
    """
    out = np.array(out).reshape(-1, 3)
    ids1 = np.unique(out[:, 0])
    ids2 = np.unique(out[:, 1])
    matches = []
    dist = np.zeros((len(ids1), len(ids2)), dtype=np.float32)
    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            cosim = out[(out[:, 0] == id1) & (out[:, 1] == id2)][:, 2]
            cosim_loss = 1 - cosim / 100
            det1 = dets1[dets1[:, 2] == id1][0]
            det2 = dets2[dets2[:, 2] == id2][0]
            iou_loss = 1 - da.get_iou(det1[3:7], det2[3:7])
            dist[i, j] = 0.9 * cosim_loss + 0.1 * iou_loss
    row_inds, col_inds = linear_sum_assignment(dist)
    for row_ind, col_ind in zip(row_inds, col_inds):
        matches.append([ids1[row_ind], ids2[col_ind]])
    # TODO something here to distiguish low quality ass and multi dets
    # tricky for multiple dets, low quality. I cover mis det in unmatched
    return matches


def get_occluded_matches_per_group(features1, features2, bbs1, bbs2):
    """
    inputs:
        image_path: Path
        vid_name: str | int
        bbs1, bbs2: np.ndarray
            detections
    output: list[list[int, int]]
        The values are the detection ids.
    """
    out = cos_sim(features1, features2, bbs1, bbs2)
    matches = get_cosim_matches_per_group(out, bbs1, bbs2)
    return matches


def get_bboxes(dets: np.ndarray, det_ids):
    """
    inputs:
        dets: np.ndarray
        det_ids: tuple[int] | list[int]
            The values are the detection ids.
    output: np.ndarray
        list of selected detections
    """
    bbs = []
    for id_ in det_ids:
        bbs.append(dets[(dets[:, 2] == id_)])
    bbs = np.concatenate(bbs, axis=0)
    return bbs


def get_occluded_matches(dets1, dets2, matching_groups, features1, features2):
    """
    inputs:
        dets1, dets2: np.ndarray
        matching_groups: dict[tuple, tuple]
        image_path: Path
        vid_name: str | int
    output: list[tuple[int, int]]
        The values are the detection ids
    """
    occluded_matches = []
    for group1, group2 in matching_groups.items():
        if (not group1) or (not group2):
            continue
        bbs1 = get_bboxes(dets1, group1)
        bbs2 = get_bboxes(dets2, group2)
        cosim_matches_group = get_occluded_matches_per_group(
            features1, features2, bbs1, bbs2
        )
        occluded_matches.extend(
            [tuple(cosim_match_group[:2]) for cosim_match_group in cosim_matches_group]
        )
    return occluded_matches


def remove_values_from_dict(original_dict, remove_values):
    """
    Remove specified values from the values of a dictionary.

    Parameters
    ----------
    original_dict : dict[tuple[int], tuple[int]]
        The original dictionary with tuples as keys and tuples as values.
    remove_values : list[int]
        A list of values to be removed from the values of the dictionary.

    Returns
    -------
    dict
        A new dictionary with the same keys as `original_dict` but with modified values.
        The modified values exclude the specified `remove_values`.

    Examples
    --------
    >>> original = {(6, 7): (5, 3), (15, 17, 18): (2, 7, 1)}
    >>> remove_values = [3, 1, 2]
    >>> remove_values_from_dict(original, remove_values)
    {(6, 7): (5,), (15, 17, 18): (7,)}
    """
    final_dict = {}
    for key, values in original_dict.items():
        modified_values = tuple(value for value in values if value not in remove_values)
        final_dict[key] = modified_values
    return final_dict


def get_matches(dets1, dets2, features1, features2, **kwargs):
    """
    inputs:
        dets1, dets2: np.ndarray
            dets1: can be the (predicted) last detections of tracklets or the image detections
            dets2: is the image detections
    output: list[tuple[int, int]]
        The values are the detection ids
    """
    occluded1 = get_occluded_dets(dets1, **kwargs)
    occluded2 = get_occluded_dets(dets2, **kwargs)
    # occluded1 = get_occluded_dets2(dets1)
    # occluded2 = get_occluded_dets2(dets2)
    matching_groups = find_match_groups(dets1, dets2, occluded1, occluded2)
    n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

    # discard the remove ids: In this way, removed det ids will not ended up in not occluded.
    # Since Hungarian matching based on iou will not be always correct.
    r_dids2 = identify_nms_dids(dets2, thrs_iou, thrs_inside)
    matching_groups = remove_values_from_dict(matching_groups, r_dids2)

    # Stage 1: Hungarian matching on non occluded detections
    n_occluded_matches = get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2)
    # return n_occluded_matches

    # Stage 2: Cos similarity of concatenated embeddings
    occluded_matches = get_occluded_matches(
        dets1, dets2, matching_groups, features1, features2
    )

    return n_occluded_matches + occluded_matches


def handle_tracklets(dets1, dets2, matches, trks, u_dids=[]):
    """
    matched and unmatched detection ids are handled here.
    unmached (tracklets: inactive track but keept, dets: new track)
    inputs:
        dets1, dets2: np.ndarray
            dets1 is the last dets of the tracklets.
            dets2 is from image.
        matches: list[tuple[int, int]]
            This is a list of matched det_id, where first is for dets1, and the second is for dets2
        trks: np.ndary
            tracks
        u_tids, u_dis: list[int]
            list of undetermined track/detection ids
    output:
    """

    did2tid = dict()
    tid = max(trks[:, 0]) + 1

    # for matches
    for match in matches:
        did1, did2 = match
        det1 = dets1[dets1[:, 2] == did1][0]
        det2 = dets2[dets2[:, 2] == did2][0]
        det2[0] = det1[0]
        det2 = np.concatenate((det2, [0, 1]))
        # last columns: dq, tq, ts (det quality, track quality, track score)
        trks = np.concatenate((trks, det2[None]), axis=0)
        did2tid[did2] = did1

    # Remove bad tracks: track is born in previous frame but then it is not
    # matched here. So it will be removed. N.B. previous step for matches should
    # be done before. Otherwise, the track length (frequency) is not 2.
    # N.B. In DeepMOT, for track birth, track is born if detections appear in 3
    # consecutive frames and have at least .3 IOU overlap. (maybe TODO)
    counter = Counter(trks[:, 0])
    freqs = np.array(list(counter.values()))
    vals = np.array(list(counter.keys()))
    bad_tids = list(vals[np.where(freqs == 1)[0]])
    for bad_tid in bad_tids:
        ind = np.where(trks[:, 0] == bad_tid)[0][0]
        trks = np.delete(trks, ind, axis=0)

    # for inactive tracks
    dids1 = dets1[:, 2]
    matched1 = [match[0] for match in matches]
    unmatched1 = set(dids1).difference(matched1)
    for did1 in unmatched1:
        det1 = dets1[dets1[:, 2] == did1][0]
        ind = np.where((trks[:, 0] == det1[0]) & (trks[:, 1] == det1[1]))[0]
        trks[ind, 13] = 2  # dq, tq, ts

    # for new track
    dids2 = dets2[:, 2]
    matched2 = [match[1] for match in matches]
    unmatched2 = set(dids2).difference(matched2 + u_dids)
    for did2 in unmatched2:
        det2 = dets2[dets2[:, 2] == did2][0]
        det2[0] = tid
        det2 = np.concatenate((det2, [0, 1]))
        # last columns: dq, tq, ts (det quality, track quality, track score)
        trks = np.concatenate((trks, det2[None]), axis=0)
        did2tid[did2] = tid
        tid += 1

    return trks, did2tid, bad_tids


def get_last_dets_tracklets(tracks):
    """
    Get the last detections and the det id changed to track id.
    The killed tracks are excluded.
    The output detections have det ids replaced by track ids.

    input:
        tracks: np.ndarray
            with tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, ts
    output:
        np.ndarray
            same as input.
    """
    dets = []
    track_ids = np.unique(tracks[:, 0])
    for tid in track_ids:
        tracklet = tracks[tracks[:, 0] == tid]
        frame_number = max(tracklet[:, 1])
        det = tracklet[tracklet[:, 1] == frame_number][0]
        if det[13] != 3:  # track status (ts) is killed
            # change det id to track id. If not changed, there might be multiple of same det id.
            det[2] = det[0]
            dets.append(det)
    return np.array(dets).astype(np.int64)


def kill_tracks(tracks, c_frame_number, thrs):
    """
    If the track is inactive for more than thrs, it will get stop status.
    The track status in tracks is changed here.

    input:
        c_frame_number: int
            The current frame number
        thrs: int
            Threshould for maximum number of inactive frames. It is step * factor. I caculated 50.
    """
    killed_tids = []
    last_dets = get_last_dets_tracklets(tracks)
    for det in last_dets:
        if c_frame_number - det[1] > thrs:
            ind = np.where((tracks[:, 0] == det[0]) & (tracks[:, 1] == det[1]))[0][0]
            tracks[ind, 13] = 3
            killed_tids.append(tracks[ind, 0])
    return killed_tids


def get_features_from_memory(memory, track_ids):
    return {track_id: memory[track_id] for track_id in track_ids}


def update_memory(memory, features2, matches, did2tid, bad_tids, u_dids=[]):
    """
    did2tid: dict(int)
        map detection id to track id
    bad_tids: list
        removed track ids
    u_dids: list[int]
        list of undetermined detection ids
    """
    # TODO killed tracks should be removed
    _ = [memory.pop(tid) for tid in bad_tids]
    for match in matches:
        tid1, did2 = match
        memory[tid1] = deepcopy(features2[did2])

    dids2 = features2.keys()
    matched2 = [match[1] for match in matches]
    unmatched2 = set(dids2).difference(matched2 + u_dids)
    for did2 in unmatched2:
        tid = did2tid[did2]
        memory[tid] = deepcopy(features2[did2])

    return memory


def discard_bad_matches(matches, disps, disp_thrs):
    """ "
    inputs
        matches: list[tuple[int,int]]
            list of matched (track_id, det_id)
        disps: dic[int]
            disps[trarck_id] = [displacements_x, displacement_y]
    outputs
        same as inputs
    """
    # TODO bug: if disps is empty the remove_tids removes all matches
    disps = {
        tid: disp
        for tid, disp in disps.items()
        if (abs(disp[0]) <= disp_thrs) & (abs(disp[1]) <= disp_thrs)
    }
    tids = [match[0] for match in matches]
    remove_tids = set(tids).difference(disps.keys())
    matches = [item for item in matches if item[0] not in remove_tids]
    return matches, disps


def calculate_displacements(trks, dets2, matches, step, disps):
    """ "
    NB. displacemnents are always between consecutive frames (consecutive by step)

    inputs
        trks: np.ndarray
            tracks
        dets2: np.ndarray
        matches: list[tuple[int,int]]
            list of matched (track_id, det_id)
    outputs
        dict[trarck_id] = [displacements_x, displacement_y]
    """
    # Use orginal dets1 instead of predicted dets1
    dets1 = get_last_dets_tracklets(trks)

    round_func = lambda x: np.int64(round(x))
    for match in matches:
        did1, did2 = match
        det1 = dets1[dets1[:, 2] == did1][0]
        det2 = dets2[dets2[:, 2] == did2][0]
        dtime = round((det2[1] - det1[1]) / step)  # dframe / step
        disps[did1] = [
            round_func((det2[7] - det1[7]) / dtime),
            round_func((det2[8] - det1[8]) / dtime),
        ]
    tids = dets1[:, 0].copy()
    matched_tids = [match[0] for match in matches]
    # inactive from beginning are removed in handle_tracklets.
    inactive_tids = list(set(tids).difference(matched_tids))
    removed_tids = set(disps.keys()).difference(matched_tids + inactive_tids)
    _ = [disps.pop(removed_tid, None) for removed_tid in removed_tids]
    return disps


def predict_locations(dets1, disps, current_frame, step):
    """
    disps already contains inactive disps as well, which are calculated in calculate_displacements.
    disps doesn't contain killed tracks.
    """
    if not disps:
        return dets1

    track_ids = dets1[:, 0].copy()
    for track_id in track_ids:
        ind = np.where(dets1[:, 0] == track_id)[0][0]
        frame_number = dets1[ind, 1]
        dtime = (current_frame - frame_number) / step
        # it should be resolved by track_birth: for inactive but not tracked
        disp_x, disp_y = disps.get(track_id, [0, 0])
        dets1[ind, [3, 5, 7]] += int(round(disp_x * dtime))
        dets1[ind, [4, 6, 8]] += int(round(disp_y * dtime))
    return dets1


def multistage_track(
    video_file,
    det_path,
    vid_name,
    start_frame,
    end_frame,
    step,
):
    det_is_array = False
    print(det_path, type(det_path), det_path.is_file())
    if det_path.is_file():
        det_is_array = True
        tracks = da.load_tracks_from_mot_format(det_path)
        tracks[:, 2] = tracks[:, 0]  # dets or tracks used as dets
        tracks[:, 0] = -1
        if tracks.shape[1] == 11:
            # detection score is 100, since mot format comes from manual annotations
            tracks = np.concatenate(
                (tracks, 100 * np.ones(len(tracks), dtype=np.int64)[:, None]), axis=1
            )

    kwargs = get_model_args()  # TODO ugly
    # DEBUG = False
    # if DEBUG:
    #     start_frame, end_frame, format = 2432, 3112, "06d"
    #     # tracks = da.load_tracks_from_mot_format(main_path / "ms_tracks.zip")
    #     tracks = np.loadtxt(main_path / "ms_tracks.txt", dtype=np.int64, delimiter=",")
    #     trks = deepcopy(tracks[tracks[:, 1] <= start_frame])
    #     # Hack to reproduce result. In this stage track is not killed but later on, the
    #     # txt file save the final one.
    #     # ind = np.where((trks[:,0]==13) & (trks[:,1]==start_frame))[0][0]
    #     # trks[ind, 13] = 2
    #     if trks.shape[1] == 11:
    #         extension = np.zeros((len(trks), 3), dtype=np.int64)
    #         extension[:, 2] = 1
    #         trks = np.concatenate((trks, extension), axis=1)
    # else:
    #     start_frame, end_frame, format = 0, 3112, "06d"
    #     trks = None
    trks = None

    # tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

    stop_thrs = step * 30
    vc = cv2.VideoCapture(str(video_file))
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, image2 = vc.read()
        if frame_number % step != 0:
            continue

        im_height, im_width = image2.shape[:2]
        if det_is_array:
            dets2 = tracks[tracks[:, 1] == frame_number]
        else:
            dets2 = da.get_detections_array(
                det_path / f"{vid_name}_{frame_number + 1}.txt",
                im_width,
                im_height,
                frame_number,
            )

        clip_bboxs(dets2, im_height, im_width)
        dets2_orig = dets2.copy()
        dets2 = non_max_sup(dets2, thrs_iou, thrs_inside)
        det_ids = dets2[:, 2].copy()
        features2 = calculate_deep_features(det_ids, dets2, image2, **kwargs)

        if trks is None:
            disps = {}
            memory = deepcopy(features2)
            trks = dets2.copy().astype(np.int64)
            trks[:, 0] = trks[:, 2]
            extension = np.repeat(np.array([[0, 1]]), len(trks), axis=0)
            # last columns: dq, tq, ts (det quality, track quality, track score)
            trks = np.concatenate((trks, extension), axis=1)
            continue
        else:
            killed_tids = kill_tracks(trks, frame_number, stop_thrs)
            dets1 = get_last_dets_tracklets(trks)
            track_ids = dets1[:, 0].copy()
            features1 = get_features_from_memory(memory, track_ids)

        if False:  # frame_number >= 1968:
            from pathlib import Path

            from tracking import visualize

            # vid_name, step, folder = 2, 8, "240hz"
            # main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
            # image_path = main_path / "images"
            # im1 = cv2.imread(
            #     str(image_path / f"{vid_name}_frame_{frame_number - step:06d}.jpg")
            # )
            # im2 = cv2.imread(
            #     str(image_path / f"{vid_name}_frame_{frame_number:06d}.jpg")
            # )
            visualize.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], image1)
            visualize.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], image2)

        dets1 = predict_locations(dets1, disps, frame_number, step)
        clip_bboxs(dets1, im_height, im_width)
        matches = get_matches(
            dets1, dets2_orig, features1, features2, close_dist_thrs=close_dist_thrs
        )
        disps = calculate_displacements(trks, dets2, matches, step, disps)
        matches, disps = discard_bad_matches(matches, disps, disp_thrs)

        # TODO: track rebirth: only if tracked for few frames. Get different status.
        # Either in handle_tracklets or other function. This is to tackle duplicate issues.
        u_dids = []  # list(set(chain(*get_occluded_dets(dets2))))
        trks, did2tid, bad_tids = handle_tracklets(dets1, dets2, matches, trks, u_dids)
        memory = update_memory(
            memory, features2, matches, did2tid, bad_tids + killed_tids, u_dids
        )

        image1 = image2.copy()

        # # save intermediate results
        # dets = get_last_dets_tracklets(trks)
        # image = cv2.imread(
        #     str(image_path / f"{vid_name}_frame_{frame_number2:06d}.jpg")
        # )
        # visualize.save_image_with_dets(
        #     image_path / "ms_tracks_inter", vid_name, dets, image
        # )

    return trks


def ultralytics_detect_video(
    video_file,
    start_frame,
    end_frame,
    step,
    det_checkpoint,
):
    """
    output:
        track: np.ndarray
            with tid=-1, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, ts
    N.B. I use the track format here. The last two colums (track quality and track status are not here.)
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 8), dtype=np.int64)
    vc = cv2.VideoCapture(video_file.as_posix())
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, image = vc.read()
        # image = cv2.imread(str(image_path / f"{vid_name}_frame_{frame_number:06d}.jpg"))
        if frame_number % step != 0:
            continue

        results = model(image, verbose=False)

        confs = np.round(100 * np.array(results[0].boxes.conf.cpu())[:, None])
        xyxy = results[0].boxes.xyxy.cpu()
        track_ids = -np.ones(len(xyxy))[:, None]
        fns = frame_number * np.ones(len(xyxy))[:, None]
        det_ids = np.arange(len(xyxy))[:, None]
        dets = np.concatenate((track_ids, fns, det_ids, xyxy, confs), axis=1).astype(
            np.int64
        )
        trks = np.concatenate((trks, dets), axis=0)

        # visualize.save_image_with_dets(
        #     main_path / f"{track_method}_tracks_inter", vid_name, dets, image
        # )
        # visualize.plot_detections_in_image(dets[:, [0,3,4,5,6]], image)
    cen_whs = np.array([list(da.cen_wh_from_tl_br(*item[3:7])) for item in trks])
    trks = np.concatenate((trks[:, :7], cen_whs, trks[:, 7:8]), axis=1).astype(np.int64)
    return trks


def ultralytics_track_video(
    video_file,
    start_frame,
    end_frame,
    step,
    det_checkpoint,
    config_file="botsort.yaml",
):
    """
    input:
        config_file: Path|None
            if None, botsort.yaml file is used. The options are botsort and bytetrack.

    output:
        track: np.ndarray
            with tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, ts
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 7), dtype=np.int64)
    vc = cv2.VideoCapture(video_file.as_posix())
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, image = vc.read()
        if frame_number % step != 0:
            continue
        results = model.track(image, persist=True, tracker=config_file, verbose=False)

        if not results[0].boxes.is_track:
            continue
        xyxy = results[0].boxes.xyxy
        track_ids = results[0].boxes.id[:, None]
        fns = frame_number * np.ones(len(xyxy))[:, None]
        dets = np.concatenate((track_ids, fns, track_ids, xyxy), axis=1).astype(
            np.int64
        )
        trks = np.concatenate((trks, dets), axis=0)

        # visualize.save_image_with_dets(
        #     main_path / f"{track_method}_tracks_inter", vid_name, dets, image
        # )
        # visualize.plot_detections_in_image(dets[:, [0,3,4,5,6]], image)
    cen_whs = np.array([list(da.cen_wh_from_tl_br(*item[3:])) for item in trks])
    trks = np.concatenate((trks, cen_whs), axis=1).astype(np.int64)
    return trks


# vid_name, step, folder = 2, 8, "240hz"
# main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
# start_frame, end_frame, format = 0, 25, "06d"
# track_method = "botsort"
# image_folder = "images_e1"
# save_name = f"{track_method}_8"
# det_checkpoint = Path("/home/fatemeh/Downloads/fish/best_model/det_best_bgr29.pt")
# config_file = Path(f"/home/fatemeh/Downloads/fish/configs/{track_method}.yaml")
# image_path = main_path / image_folder

# 1. s1: hungarian dist&iou on high quality dets no overlap (I have no_ovelap version)
# 2. s2: hungarian agg cossim on coverlap -> low quality ass (either low value or multiple detection)
# 3. s3: hungarian dist&iou on low quality dets
# N.B. killed tracks are excluded in getting last dets of tracklets
# ts: track states. 1- new/tracked, 2- untracked/inactive, 3- killed/stopped
# good video for debugging
#   vid_name, step, folder = 2, 8, "240hz"
#   main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
# other videos for debugging: 2, 184, "240hz", 6, 8, "240hz"  # 0, 1, "30hz"
# N.B. I can't fairly debug. Because killed track is automatically removed, where in
# real case it still has status of inactive.
# tracks = da.compute_tracks(main_path/"yolo", "2", 1920, 1080, 0, 3112, 8)
# tracks = da._reindex_tracks(da._remove_short_tracks(tracks))
# trks = da.make_array_from_tracks(tracks)
# visualize.save_images_with_tracks(main_path/"hung", main_path/"vids/2.mp4", trks, 0, 3112, 8, '06d')
# 1000 (32x32) -> 3 second for calculate_deep_features
# TODO gt for track as option
# TODO compare ms_track, hungerian, bytetrack, botsort
# TODO save as images or video should be combined
# TODO low quality det
# TODO very short tracks: caused by mismatch.
