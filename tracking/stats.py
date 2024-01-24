from pathlib import Path

import motmetrics as mm
import numpy as np

from tracking.data_association import get_iou


def get_gt_object_match(atracks, annos, track_id, frame_number, thres=20, min_iou=0.1):
    det_gt = annos[(annos[:, 0] == track_id) & (annos[:, 1] == frame_number)][0]

    candidates = atracks[
        (atracks[:, 1] == frame_number)
        & (
            (
                (abs(atracks[:, 3] - det_gt[3]) < thres)
                & (abs(atracks[:, 4] - det_gt[4]) < thres)
            )
            | (
                (abs(atracks[:, 5] - det_gt[5]) < thres)
                & (abs(atracks[:, 6] - det_gt[6]) < thres)
            )
        )
    ]
    if len(candidates) == 0:
        return det_gt, None
    ious = []
    dets = []
    for det in candidates:
        ious.append([det[0], get_iou(det_gt[3:7], det[3:7])])
        dets.append(det)
    ious = np.array(ious)
    iou_max = max(ious[:, 1])
    if iou_max < min_iou:
        return det_gt, None
    track_id = ious[ious[:, 1] == iou_max][0, 0]
    det = [det for det in dets if det[0] == track_id][0]
    return det_gt, det


def get_stats_for_frame(annos, atracks, frame_number):
    tp = fp = fn = 0
    gt_track_ids = np.unique(annos[annos[:, 1] == frame_number, 0])
    matched_ids = []
    for gt_track_id in gt_track_ids:
        det1, det2 = get_gt_object_match(
            atracks, annos, gt_track_id, frame_number, thres=20, min_iou=0.1
        )
        if det2 is None:
            fn += 1
        else:
            tp += 1
            matched_ids.append([det1[0], det2[0]])
    matched_ids = np.array(matched_ids).astype(np.int64)
    track_ids = np.unique(atracks[atracks[:, 1] == frame_number, 0])
    diff_ids = set(track_ids).difference(set(matched_ids[:, 1]))
    fp = len(diff_ids)

    # gt_diff_ids = set(gt_track_ids).difference(set(matched_ids[:, 0]))
    # print(f"matched ids tracks:\n{matched_ids}")
    # print(f"diff ids tracks:\n{diff_ids}")
    # print(f"diff ids gt:\n{gt_diff_ids}")
    return tp, fp, fn


def get_stats_for_track(annos, atracks, track_id):
    tp = fp = fn = 0
    frame_numbers = annos[annos[:, 0] == track_id, 1]
    matched_ids = []
    for frame_number in frame_numbers:
        det1, det2 = get_gt_object_match(
            atracks, annos, track_id, frame_number, thres=20, min_iou=0.1
        )
        if det2 is None:
            fn += 1
        else:
            tp += 1
            matched_ids.append([det1[0], det2[0], frame_number])
    if len(matched_ids) == 0:
        return tp, -1, fn, -1, -1, matched_ids
    matched_ids = np.array(matched_ids).astype(np.int64)

    unique_ids = np.unique(np.sort(matched_ids[:, 1]))
    freq, _ = np.histogram(
        matched_ids[:, 1], bins=np.hstack((unique_ids, unique_ids[-1] + 1))
    )
    main_track_id = unique_ids[freq == max(freq)][0]
    no_switch_ids = len(matched_ids[matched_ids[:, 1] != main_track_id, 1])
    no_unique_ids = len(unique_ids)

    # here fn is calculated based on dominant track_id.
    main_track_frame_numbers = atracks[atracks[:, 0] == main_track_id, 1]
    matched_main_track_frame_numbers = matched_ids[
        matched_ids[:, 1] == main_track_id, 2
    ]
    fp = len(set(main_track_frame_numbers).difference(matched_main_track_frame_numbers))
    return tp, fp, fn, no_switch_ids, no_unique_ids, matched_ids


def get_stats_for_tracks(annos, atracks):
    stats = []
    for track_id in np.unique(annos[:, 0]):
        tp, fp, fn, sw, uid, _ = get_stats_for_track(annos, atracks, track_id)
        stats.append([track_id, tp, fp, fn, sw, uid])
    return np.array(stats).astype(np.int64)


def compare_one_file(tr_gt_file, tr_pr_file, save_file):
    """
    Examples
    --------
    >>> exp = "ms_exp1"
    >>> gt_path = Path("/home/fatemeh/Downloads/fish/mot_data/challenge/tr_gt/8_1.txt")
    >>> track_path = Path("/home/fatemeh/Downloads/fish/mot_data/challenge/tr_pred/8_1_ms_exp1.txt")
    >>> save_file = gt_path.parent.parent / f"{exp}.txt"
    >>> compare_one_file(gt_path, track_path, save_file, exp)
    """
    gt = np.loadtxt(tr_gt_file, delimiter=",")
    pred = np.loadtxt(tr_pr_file, delimiter=",")
    sts = get_stats_for_tracks(gt, pred)

    id_sw = sum([i[-1] - 1 for i in sts if i[-1] != -1])
    n_tracks = sts.shape[0]
    n_frames = len(np.unique(gt[:, 1]))
    n_objects = gt.shape[0]
    n_no_tracks = sum([1 for i in sts if i[-1] == -1])
    vid_name = tr_gt_file.stem

    with open(save_file, "a") as rfile:
        rfile.write(
            f"{vid_name},{n_frames},{n_tracks},{n_objects},{id_sw},{n_no_tracks}\n"
        )


def compare_files(gt_path, track_path, save_file, exp):
    """
    Examples
    --------
    >>> exp = "ms_exp1"
    >>> gt_path = Path("/home/fatemeh/Downloads/fish/mot_data/challenge/txt_gt")
    >>> track_path = Path(f"/home/fatemeh/Downloads/fish/mot_data/challenge/txt_{exp}")
    >>> save_file = gt_path.parent / f"{exp}.txt"
    >>> compare_files(gt_path, track_path, save_file, exp)
    """
    for gt_file in gt_path.glob("*txt"):
        pr_file = track_path / f"{gt_file.stem}_{exp}.txt"
        compare_one_file(gt_file, pr_file, save_file)


def motMetricsEnhancedCalculator(gtSource, tSource, max_iou=0.9):
    """
    Examples
    --------
    >>> gt_file = "/home/fatemeh/Downloads/fish/mot_data/challenge/ch_gt/8_1.txt"
    >>> track_file = "/home/fatemeh/Downloads/fish/mot_data/challenge/ch_ms_exp1/8_1_ms_exp1.txt"
    >>> summary, strsummary = motMetricsEnhancedCalculator(gt_file, track_file, 0.9)
    """
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=",")

    # load tracking output
    t = np.loadtxt(tSource, delimiter=",")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=max_iou
        )  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(),
            t_dets[:, 0].astype("int").tolist(),
            C,
        )

    mh = mm.metrics.create()

    metrics = [i.split("|")[0] for i in mh.list_metrics_markdown().split("\n")[2:-1]]
    summary = mh.compute(acc, metrics=metrics, name="acc")
    # summary.to_dict()['num_objects']
    strsummary = mm.io.render_summary(summary, namemap=dict(zip(metrics, metrics)))

    print(strsummary)
    return summary, strsummary
