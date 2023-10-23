# - use Hungarian for each not occluded
# - use Cosim for each occluded group

from copy import deepcopy
from itertools import chain, combinations
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO

from tracking import data_association as da
from tracking import visualize

np.random.seed(1000)

# TODO put it in get_model_args
layers = ["conv1", "layer1", "layer2", "layer3"]
im_width, im_height = 1920, 1080


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
        The values are the detection ids.

    output e.g. [[10, 15, 17], [21, 29]]
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


def merge_overlapping_keys_values(input):
    """
    merge overlapping keys

    input, output: dic[tuple[int, ...], tuple[int, ...]]

    input output e.g. {(6, 7): (6, 7), (15, 17): (15, 17)}
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


def find_match_groups(dets1, dets2, occluded1, occluded2):
    # TODO efficient implementation
    """
    inputs:
        dets1, dets2: np.ndarray
        occluded1, occluded2: list[list[int]]
    output: dic[tuple[int, ...], tuple[int, ...]]
        The values are the detection ids.

    output e.g. {(6, 7): (6, 7), (15, 17): (15, 17)
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
        matching_groups[group1] = group2
    matching_groups = merge_overlapping_keys_values(matching_groups)
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
            loc_loss = np.linalg.norm([det2[7] - det1[7], det2[8] - det1[8]])
            dist[i, j] = iou_loss + loc_loss
    row_ind, col_ind = linear_sum_assignment(dist)
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


def cos_sim(features1, features2, bbs1, bbs2, **kwargs):
    """
    inputs:
        image_path: Path
        vid_name: str | int
        bbs1, bbs2: np.ndarray
    output: list[int]
        The values are the detection ids.

    e.g. output [44, 44, 83, 44, 15, 81, 15, 44, 82, 15, 15, 85]
    """
    bbs1 = np.array([bbox_enlarge(bb) for bb in bbs1])
    bbs2 = np.array([bbox_enlarge(bb) for bb in bbs2])
    # w, h = max(max(bbs1[:, -2]), max(bbs2[:, -2])), max(
    #     max(bbs1[:, -1]), max(bbs2[:, -1])
    # )

    clip_bboxs(bbs1, im_height, im_width)
    clip_bboxs(bbs2, im_height, im_width)

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


def get_cosim_matches_per_group(out):
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
            dist[i, j] = 1 - cosim / 100
    row_inds, col_inds = linear_sum_assignment(dist)
    for row_ind, col_ind in zip(row_inds, col_inds):
        cosim = int(round((1 - dist[row_ind, col_ind]) * 100))
        matches.append([ids1[row_ind], ids2[col_ind], cosim])
    # TODO something here to distiguish low quality ass and multi dets
    # tricky for multiple dets, low quality. I cover mis det in unmatched
    return matches


def get_occluded_matches_per_group(features1, features2, bbs1, bbs2, **kwargs):
    """
    inputs:
        image_path: Path
        vid_name: str | int
        bbs1, bbs2: np.ndarray
    output: list[list[int, int]]
        The values are the detection ids.
    """
    out = cos_sim(features1, features2, bbs1, bbs2, **kwargs)
    matches = get_cosim_matches_per_group(out)
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


def get_occluded_matches(dets1, dets2, matching_groups, features1, features2, **kwargs):
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
            features1, features2, bbs1, bbs2, **kwargs
        )
        occluded_matches.extend(
            [tuple(cosim_match_group[:2]) for cosim_match_group in cosim_matches_group]
        )
    return occluded_matches


def get_matches(dets1, dets2, features1, features2, **kwargs):
    """
    inputs:
        dets1, dets2: np.ndarray
            dets1: can be the (predicted) last detections of tracklets or the image detections
            dets2: is the image detections
    output: list[tuple[int, int]]
        The values are the detection ids
    """
    occluded1 = get_occluded_dets(dets1)
    occluded2 = get_occluded_dets(dets2)
    matching_groups = find_match_groups(dets1, dets2, occluded1, occluded2)
    n_occluded1, n_occluded2 = get_not_occluded(dets1, dets2, matching_groups)

    # Stage 1: Hungarian matching on non occluded detections
    n_occluded_matches = get_n_occluded_matches(dets1, dets2, n_occluded1, n_occluded2)

    # Stage 2: Cos similarity of concatenated embeddings
    occluded_matches = get_occluded_matches(
        dets1, dets2, matching_groups, features1, features2, **kwargs
    )

    return n_occluded_matches + occluded_matches


def handle_tracklets(dets1, dets2, matches, trks):
    """
    matched and unmatched detection ids are handled here.
    unmached (tracklets: inactive track but keept, dets: new track)
    inputs:
        dets1, dets2: np.ndarray
            dets1 is the last dets of the tracklets.
            dets2 is from image.
        matches: list[tuple[int, int]]
            This is a list of matched det_id, where first is for dets1, and the second is for dets2
    output:
    """

    did2tid = dict()
    tid = max(trks[:, 0]) + 1

    for match in matches:
        did1, did2 = match
        det1 = dets1[dets1[:, 2] == did1][0]
        det2 = dets2[dets2[:, 2] == did2][0]
        det2[0] = det1[0]
        det2 = np.concatenate((det2, [0, 0, 1]))
        trks = np.concatenate((trks, det2[None]), axis=0)
        did2tid[did2] = did1

    dids1 = dets1[:, 2]
    matched1 = [match[0] for match in matches]
    unmatched1 = set(dids1).difference(matched1)
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
        det2 = np.concatenate((det2, [0, 0, 1]))
        trks = np.concatenate((trks, det2[None]), axis=0)
        did2tid[did2] = tid
        tid += 1

    return trks, did2tid


def get_last_dets_tracklets(tracks):
    """
    Get the last detections and the det id changed to track id.
    The killed tracks are excluded.
    The output detections have det ids replaced by track ids.

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
        if det[13] != 3:  # track status (ts) is killed
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
            tracks[ind, 13] = 3


def get_features_from_memory(memory, track_ids):
    return {track_id: memory[track_id] for track_id in track_ids}


def update_memory(memory, features2, matches, did2tid):
    for match in matches:
        did1, did2 = match
        memory[did1] = features2[did2]

    dids2 = features2.keys()
    matched2 = [match[1] for match in matches]
    unmatched2 = set(dids2).difference(matched2)
    for did2 in unmatched2:
        tid = did2tid[did2]
        memory[tid] = features2[did2]

    return memory


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

    stop_thr = step * 50
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

        det_ids = dets2[:, 2].copy()
        features2 = calculate_deep_features(det_ids, dets2, image2, **kwargs)

        if trks is None:
            memory = deepcopy(features2)
            trks = dets2.copy().astype(np.int64)
            trks[:, 0] = trks[:, 2]
            extension = np.repeat(
                np.array([[0, 0, 1]]), len(trks), axis=0
            )  # ts, dq, tq
            trks = np.concatenate((trks, extension), axis=1)
            continue
        else:
            dets1 = get_last_dets_tracklets(trks)
            kill_tracks(trks, dets2, frame_number, stop_thr)
            dets1 = get_last_dets_tracklets(trks)
            track_ids = dets1[:, 0].copy()
            features1 = get_features_from_memory(memory, track_ids)

        # if DEBUG:
        #     im1 = cv2.imread(
        #         str(image_path / f"/{vid_name}_frame_{frame_number1:06d}.jpg")
        #     )
        #     im2 = cv2.imread(
        #         str(image_path / f"{vid_name}_frame_{frame_number2:06d}.jpg")
        #     )
        #     visualize.plot_detections_in_image(dets1[:, [2, 3, 4, 5, 6]], im1)
        #     plt.show(block=False)
        #     visualize.plot_detections_in_image(dets2[:, [2, 3, 4, 5, 6]], im2)
        #     plt.show(block=False)

        # TODO clip_bbox in here.
        matches = get_matches(dets1, dets2, features1, features2, **kwargs)

        trks, did2tid = handle_tracklets(dets1, dets2, matches, trks)
        memory = update_memory(memory, features2, matches, did2tid)

        # # save intermediate results
        # dets = get_last_dets_tracklets(trks)
        # image = cv2.imread(
        #     str(image_path / f"{vid_name}_frame_{frame_number2:06d}.jpg")
        # )
        # visualize.save_image_with_dets(
        #     image_path / "ms_tracks_inter", vid_name, dets, image
        # )

    return trks


def ultralytics_detect(
    image_path,
    vid_name,
    start_frame,
    end_frame,
    step,
    det_checkpoint,
):
    """ "
    input:

    output:
        track: np.ndarray
            with tid=-1, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 7), dtype=np.int64)
    for frame_number in tqdm(range(start_frame, end_frame + 1, step)):
        image = cv2.imread(str(image_path / f"{vid_name}_frame_{frame_number:06d}.jpg"))
        results = model(image, verbose=False)

        xyxy = results[0].boxes.xyxy.cpu()
        track_ids = -np.ones(len(xyxy))[:, None]
        ones = frame_number * np.ones(len(xyxy))[:, None]
        det_ids = np.arange(len(xyxy))[:, None]
        dets = np.concatenate((track_ids, ones, det_ids, xyxy), axis=1).astype(np.int64)
        trks = np.concatenate((trks, dets), axis=0)

        # visualize.save_image_with_dets(
        #     main_path / f"{track_method}_tracks_inter", vid_name, dets, image
        # )
        # visualize.plot_detections_in_image(dets[:, [0,3,4,5,6]], image)
    cen_whs = np.array([list(da.cen_wh_from_tl_br(*item[3:])) for item in trks])
    trks = np.concatenate((trks, cen_whs), axis=1).astype(np.int64)
    return trks


def ultralytics_track(
    image_path,
    vid_name,
    start_frame,
    end_frame,
    step,
    det_checkpoint,
    config_file="botsort.yaml",
):
    """ "
    input:
        config_file: Path|None
            if None, botsort.yaml file is used. The options are botsort and bytetrack.

    output:
        tracks: np.ndarray
            with tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 7), dtype=np.int64)
    for frame_number in tqdm(range(start_frame, end_frame + 1, step)):
        image = cv2.imread(str(image_path / f"{vid_name}_frame_{frame_number:06d}.jpg"))
        results = model.track(image, persist=True, tracker=config_file, verbose=False)

        xyxy = results[0].boxes.xyxy
        track_ids = results[0].boxes.id[:, None]
        ones = frame_number * np.ones(len(xyxy))[:, None]
        dets = np.concatenate((track_ids, ones, track_ids, xyxy), axis=1).astype(
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


def ultralytics_detect_video(
    video_file,
    start_frame,
    end_frame,
    step,
    det_checkpoint,
):
    """ "
    input:

    output:
        track: np.ndarray
            with tid=-1, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 7), dtype=np.int64)
    vc = cv2.VideoCapture(video_file.as_posix())
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, image = vc.read()
        if frame_number % step == 0:
            results = model(image, verbose=False)

            xyxy = results[0].boxes.xyxy.cpu()
            track_ids = -np.ones(len(xyxy))[:, None]
            ones = frame_number * np.ones(len(xyxy))[:, None]
            det_ids = np.arange(len(xyxy))[:, None]
            dets = np.concatenate((track_ids, ones, det_ids, xyxy), axis=1).astype(
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
            with tid, fn, did, x, y, x, y, cx, cy, w, h, dq, tq, st
    """
    model = YOLO(det_checkpoint)

    trks = np.empty((0, 7), dtype=np.int64)
    vc = cv2.VideoCapture(video_file.as_posix())
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, image = vc.read()
        if frame_number % step == 0:
            results = model.track(
                image, persist=True, tracker=config_file, verbose=False
            )

            xyxy = results[0].boxes.xyxy
            track_ids = results[0].boxes.id[:, None]
            ones = frame_number * np.ones(len(xyxy))[:, None]
            dets = np.concatenate((track_ids, ones, track_ids, xyxy), axis=1).astype(
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


# vid_name, frame_number1, step, folder = 2, 184, 8, "240hz"
# main_path = Path(f"/home/fatemeh/Downloads/fish/in_sample_vids/{folder}")
# start_frame, end_frame, format = 0, 25, "06d"
# track_method = "botsort"
# image_folder = "images_e1"
# save_name = f"{track_method}_8"
# det_checkpoint = Path("/home/fatemeh/Downloads/fish/best_model/det_best_bgr29.pt")
# config_file = Path(f"/home/fatemeh/Downloads/fish/configs/{track_method}.yaml")
# image_path = main_path / image_folder

# trks = ultralytics_detect(
#     image_path,
#     vid_name,
#     start_frame,
#     end_frame,
#     step,
#     det_checkpoint,
# )
# da.save_tracks_to_mot_format(main_path / "test.zip", trks[:, :11])
# dets = da.load_tracks_from_mot_format(main_path / "test.zip")
# dets[:, 2] = dets[:, 0]
# dets[:, 0] = -1
# trks = multistage_track(
#     main_path,
#     image_folder,
#     vid_name,
#     start_frame,
#     end_frame,
#     step,
# )
# # trks = ultralytics_track(
# #     image_path,
# #     vid_name,
# #     start_frame,
# #     end_frame,
# #     step,
# #     det_checkpoint,
# #     config_file="botsort.yaml",
# # )
# np.savetxt(main_path / f"{save_name}.txt", trks, delimiter=",", fmt="%d")
# da.save_tracks_to_mot_format(main_path / f"{save_name}.zip", trks[:, :11])
# visualize.save_images_with_tracks(
#     main_path / save_name,
#     main_path / f"vids/{vid_name}.mp4",
#     trks,
#     start_frame,
#     end_frame,
#     step,
#     format,
# )

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
# TODO clip_bbox in multistage_track
# TODO gt for track as option
# TODO predict location (constant speed): take care of visualization/saving
# TODO compare ms_track, hungerian, bytetrack, botsort
# TODO save as images or video should be combined
# TODO low quality det
# TODO very short tracks: caused by mismatch.
# TODO (maybe) In DeepMOT, for track birth, track is born if detections appear in 3 consecutive frames and have at least .3 IOU overlap.
