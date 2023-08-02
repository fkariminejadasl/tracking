# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
import torchvision
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking import data_association as da
from tracking import visualize

np.random.seed(1000)


def merge_intersecting_items(lst):
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


def get_occluded_dets(dets):
    """
    input:
        dets np.ndarray
    output: list[list[int]]

    output e.g. [[10, 13, 17], [21, 29]]
    """

    # if 8, 9, 11, where 9, 11 intersect, 8, 11 but not 8, 9. This return two groups.
    # if 11 was a smaller number then one group of three is return. I'm not sure if I change
    # this part. -> now is changed by merge_intersecting_items
    occluded = {}
    ids = dets[:, 2]
    for did1, did2 in combinations(ids, 2):
        det1 = dets[dets[:, 2] == did1][0]
        det2 = dets[dets[:, 2] == did2][0]
        if da.get_iou(det1[3:7], det2[3:7]) > 0:
            occluded.setdefault(did1, [did1]).append(did2)
    occluded = list(occluded.values())
    return merge_intersecting_items(occluded)


def find_match_groups(dets1, dets2, occluded1, occluded2):
    """
    inputs:
        dets1, dets2: np.ndarray
        occluded1, occluded2: list[list[int]]
    output: dic[tuple[int, ...], tuple[int, ...]]

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
        group2 = tuple(sorted(set(group2)))
        values = []
        for did2 in group2:
            det2 = dets2[dets2[:, 2] == did2][0]
            for det1 in dets1:
                did1 = det1[2]
                if da.get_iou(det1[3:7], det2[3:7]) > 0:
                    values.append(did1)
        group1 = tuple(sorted(set(values)))
        if group1 not in matching_groups.keys():
            matching_groups[group1] = group2
    return matching_groups


def get_not_occluded(dets1, dets2, matching_groups):
    """
    inputs:
        dets1, dets2: np.ndarray
        matching_groups: dic[tuple[int, ...], tuple[int, ...]]
    output: tuple[set[int], set[int]]
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
    output: tuple[list[int], list[int]]
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
        n_occluded1, n_occluded2: set[int]
    output: list[tuple[int, int]]
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
        vid_name: str | int
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
            output.extend([bb1[2], bb2[2], csim])
    return output


def get_cosim_matches_per_group(out):
    """
    input:
        out: list[int]
    output: list[list[int]]

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
        vid_name: str | int
        bbs1, bbs2: np.ndarray
    output: list[tuple[int, int]]
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
        matching_groups: dict[tuple, tuple]
        main_path: Path
        vid_name: str | int
    output: list[tuple[int, int]]
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
    output: list[tuple[int, int]]
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

    # print(occluded1, n_occluded1)
    # print(occluded2, n_occluded2)
    # print(matching_groups)
    # print(n_occluded_matches)
    # print(occluded_matches)

    return n_occluded_matches + occluded_matches


def handle_tracklets(dets1, dets2, matches, trks=None):
    """
    matched and unmatched detection ids are handled here.
    unmached (tracklets: inactive track but keept, dets: new track)
    inputs:
        dets1, dets2: np.ndarray
            dets1 is either detections from image or the last dets of the tracklets.
            dets2 is from image.
        matches: list[tuple[int, int]]
            This is a list of matched det_id, where first is for dets1, and the second is for dets2
    output:
    """
    if trks is None:
        trks = np.empty(shape=(0, 14), dtype=np.int64)
        tid = 0
    else:
        tid = max(trks[:, 0]) + 1

    for match in matches:
        did1, did2 = match
        det1 = dets1[dets1[:, 2] == did1][0]
        det2 = dets2[dets2[:, 2] == did2][0]
        if det1[0] == -1:  # det from image
            det1[0] = tid
            det2[0] = tid
            det1 = np.concatenate((det1, [0, 0, 1]))  # ts, dq, tq
            det2 = np.concatenate((det2, [0, 0, 1]))
            det12 = np.stack((det1, det2), axis=0)
            trks = np.concatenate((trks, det12), axis=0)
            tid += 1
        else:  # last det of tracklet
            det2[0] = det1[0]
            det2 = np.concatenate((det2, [0, 0, 1]))
            trks = np.concatenate((trks, det2[None]), axis=0)

    dids1 = dets1[:, 2]
    matched1 = [match[0] for match in matches]
    unmatched1 = set(dids1).difference(matched1)
    if dets1[0, 0] == -1:  # det from image
        for did1 in unmatched1:
            det1 = dets1[dets1[:, 2] == did1][0]
            det1[0] = tid
            tid += 1
            det1 = np.concatenate((det1, [0, 0, 2]))  # ts, dq, tq
            trks = np.concatenate((trks, det1[None]), axis=0)
    else:  # last det of tracklet
        for did1 in unmatched1:
            det1 = dets1[dets1[:, 2] == did1][0]
            ind = np.where((trks[:, 0] == det1[0]) & (trks[:, 1] == det1[1]))[0]
            trks[ind, 13] = 2  # ts, dq, tq

    dids2 = dets2[:, 2]
    matched2 = [match[1] for match in matches]
    unmatched2 = set(dids2).difference(matched2)
    for did2 in unmatched2:
        det2 = dets2[dets2[:, 2] == did2][0]
        det2[0] = tid
        tid += 1
        det2 = np.concatenate((det2, [0, 0, 1]))
        trks = np.concatenate((trks, det2[None]), axis=0)

    return trks


def get_last_dets_tracklets(tracks):
    """
    Get the last detections and the det id changed to track id
    input:
        tracks: np.ndarray
            with tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st
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
        # change det id to track id. If not changed, there might be multiple of same det id.
        det[2] = det[0]
        dets.append(det)
    return np.array(dets).astype(np.int64)


def kill_tracks(tracks, last_dets, c_frame_number, thr=50):
    """
    If the track is inactive for more than thr, it will get stop status.
    The track status in tracks is changed here.
    input:
        last_dets: np.ndarray
            The last detections of a tracklet
        c_frame_number: int
            The current frame number
        thr: int
            Threshould for maximum number of inactive frames. It is step * factor. I caculated 50.
    """
    for det in last_dets:
        if c_frame_number - det[1] > thr:
            ind = np.where((tracks[:, 0] == det[0]) & (tracks[:, 1] == det[1]))[0][0]
            tracks[ind, 13] = 2


kwargs = get_model_args()  # TODO ugly

# 6, 16, 8, "240hz"  # 2, 184, 8, "240hz", # 0, 38, 1, "30hz"
vid_name, frame_number1, step, folder = 2, 184, 8, "240hz"
main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")

DEBUG = False
if DEBUG:
    start_frame, end_frame, format = 2264, 3112, "06d"
    tracks = da.load_tracks_from_mot_format(main_path / "ms_tracks.txt.zip")
    trks = deepcopy(tracks[tracks[:, 1] <= start_frame])
    extension = np.zeros((len(trks), 3), dtype=np.int64)
    extension[:, 2] = 1
    trks = np.concatenate((trks, extension), axis=1)

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

# """
start_frame, end_frame, format = 0, 3112, "06d"
stop_thr = step * 50
trks = None
for frame_number1 in tqdm(range(start_frame, end_frame + 1, step)):
    frame_number2 = frame_number1 + step

    if trks is None:
        dets1 = tracks[tracks[:, 1] == frame_number1]
        dets1[:, 2] = dets1[:, 0]  # TODO hack to missuse tracks for detections
        dets1[:, 0] = -1
    else:
        dets1 = get_last_dets_tracklets(trks)
        kill_tracks(trks, dets1, frame_number2, stop_thr)

    dets2 = tracks[tracks[:, 1] == frame_number2]

    if DEBUG:
        im1 = cv2.imread(
            str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg")
        )
        im2 = cv2.imread(
            str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg")
        )
        visualize.plot_detections_in_image(dets1[:, [0, 3, 4, 5, 6]], im1)
        plt.show(block=False)
        visualize.plot_detections_in_image(dets2[:, [0, 3, 4, 5, 6]], im2)
        plt.show(block=False)

    dets2[:, 2] = dets2[:, 0]  # TODO hack to missuse tracks for detections
    dets2[:, 0] = -1

    matches = get_matches(dets1, dets2, main_path, vid_name, **kwargs)
    trks = handle_tracklets(dets1, dets2, matches, trks)

da.save_tracks_to_mot_format(main_path / "ms_tracks.txt", trks[:, :11])
visualize.save_video_with_tracks_as_images(
    main_path / "ms_tracks",
    main_path / f"vids/{vid_name}.mp4",
    trks[:, :11],
    start_frame,
    end_frame,
    step,
    format,
)
# """

# TODO
# 1. s1: hungarian dist&iou on high quality dets no overlap (I have no_ovelap version)
# 2. s2: hungarian agg cossim on coverlap -> low quality ass (either low value or multiple detection)
# 3. s3: hungarian dist&iou on low quality dets
# TODO bug frame 2432: switch ids 9 to 1

# =============================
kwargs = get_model_args()
vid_name, frame_number1, step, folder = 6, 16, 8, "240hz"
main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
frame_number2 = frame_number1 + step

tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")

dets1 = tracks[tracks[:, 1] == frame_number1]
dets2 = tracks[tracks[:, 1] == frame_number2]
# missuse tracks for detections
dets1[:, 2] = dets1[:, 0]
dets2[:, 2] = dets2[:, 0]


def test_merge_intersecting_items():
    occluded = [[8, 10], [7, 9], [6, 8, 10]]
    expected = [[6, 8, 10], [7, 9]]
    new = merge_intersecting_items(occluded)
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
    groups = get_occluded_dets(dets)
    assert groups == [[0, 1, 3, 4], [2, 5]]


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
    occluded_matches = get_occluded_matches(
        dets1, dets2, matching_groups, main_path, vid_name, **kwargs
    )
    assert exp_occluded_matches == occluded_matches


def test_handle_tracklets():
    exp_trks = np.array(
        [
            [0, 184, 1, 1127, 417, 1142, 445, 1135, 431, 15, 28, 0, 0, 1],
            [0, 192, 11, 1127, 417, 1141, 447, 1134, 432, 14, 29, 0, 0, 1],
            [1, 184, 2, 1493, 452, 1510, 472, 1501, 462, 17, 20, 0, 0, 1],
            [1, 192, 12, 1493, 455, 1509, 475, 1501, 465, 17, 20, 0, 0, 1],
            [2, 184, 3, 1075, 330, 1091, 340, 1083, 335, 16, 10, 0, 0, 1],
            [2, 192, 13, 1074, 331, 1090, 341, 1082, 336, 17, 10, 0, 0, 1],
            [3, 184, 6, 1076, 499, 1098, 513, 1087, 506, 22, 13, 0, 0, 1],
            [3, 192, 16, 1076, 499, 1098, 513, 1087, 506, 22, 13, 0, 0, 1],
            [4, 184, 7, 1047, 480, 1074, 488, 1060, 484, 27, 8, 0, 0, 1],
            [4, 192, 17, 1046, 481, 1073, 489, 1059, 485, 27, 8, 0, 0, 1],
            [5, 184, 8, 1171, 584, 1201, 610, 1186, 597, 30, 27, 0, 0, 1],
            [5, 192, 18, 1171, 584, 1202, 611, 1187, 597, 31, 27, 0, 0, 1],
            [6, 184, 4, 1215, 301, 1227, 319, 1221, 310, 12, 18, 0, 0, 1],
            [6, 192, 14, 1215, 301, 1227, 319, 1221, 310, 12, 18, 0, 0, 1],
            [7, 184, 0, 1198, 448, 1217, 478, 1207, 463, 19, 30, 0, 0, 2],
            [8, 184, 5, 1211, 317, 1223, 333, 1217, 325, 13, 15, 0, 0, 2],
            [9, 192, 10, 1197, 450, 1216, 481, 1206, 466, 19, 30, 0, 0, 1],
            [10, 192, 15, 1210, 318, 1223, 333, 1217, 326, 13, 15, 0, 0, 1],
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

    matches = get_matches(dets1, dets2, main_path, vid_name, **kwargs)
    matches.remove((0, 10))
    matches.remove((5, 15))
    trks = handle_tracklets(dets1, dets2, matches)

    np.testing.assert_array_equal(trks, exp_trks)


def test_kill_tracks():
    vid_name, folder = 2, "240hz"
    main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")

    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
    extension = np.zeros((len(tracks), 3), dtype=np.int64)
    tracks = np.concatenate((tracks, extension), axis=1)

    dets1 = get_last_dets_tracklets(tracks)
    kill_tracks(tracks, dets1, 3117, 50)
    assert tracks[30193, 13] == 0
    kill_tracks(tracks, dets1, 4000, 50)
    assert tracks[30193, 13] == 2


test_merge_intersecting_items()
test_get_occluded_dets()
test_find_match_groups()
test_get_cosim_matches_per_group()
test_stage1()
test_stage2()
test_handle_tracklets()
test_kill_tracks()
print("passed")
