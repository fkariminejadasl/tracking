from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import linear_sum_assignment

from tracking import data_association as da
from tracking import postprocess as pp


# normalize curves by translating them to have the same starting point (0,0) and scaling
def normalize_curve(curve):
    curve = deepcopy(curve)
    curve -= curve.min(axis=0)
    max_length = np.max(curve.max(axis=0) - curve.min(axis=0))
    return curve / max_length if max_length > 0 else curve


# compute the sum of squared distances between curves, multiplied by curve number of points
def curve_distance(curve1, curve2):
    n_points = curve1.shape[0]
    return np.sum((curve1 - curve2) ** 2) * 1 / n_points


def cut_track_into_tracklets(track, start, end, step):
    """
    Cut a track into tracklets within a specified frame range, creating a new tracklet
    every 'step' number of frames.

    Parameters:
    - track: numpy array representing the track to be cut into tracklets.
             The expected format of the array is:
             [track_id, frame_number, det_id, x_tl, y_tl, x_br, y_br, x_cen, y_cen, width, height].
    - start: int, the starting frame number from which to begin creating tracklets.
    - end: int, the ending frame number at which to stop creating tracklets.
    - step: int, the number of frames after which to cut the track into a new tracklet.

    Returns:
    - List of tracklets: A list where each element is a numpy array representing a tracklet
                         with the same format as the input track.
    """

    tracklets = []
    for st in range(start, end + 1, step):
        en = min(st + step - 1, end)
        tracklet = track[(track[:, 1] >= st) & (track[:, 1] <= en)]
        if tracklet.size > 0:
            tracklet = deepcopy(tracklet)
            tracklet[:, 2] = st
            tracklets.extend(tracklet)
    return np.array(tracklets)


def cut_tracks_into_tracklets(tracks, start, end, step):
    tids = set(tracks[:, 0])
    tracklets = []
    for tid in tids:
        track = tracks[tracks[:, 0] == tid]
        tracklets.extend(cut_track_into_tracklets(track, start, end, step))
    return np.array(tracklets)


# fmt: off
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# fmt: on
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
# tracks1 = da.load_tracks_from_mot_format(
#     Path("/home/fatemeh/Downloads/fish/mot_data/ms_exp1/mots/129_1_ms_exp1.txt")
# )
# tracks2 = da.load_tracks_from_mot_format(
#     Path("/home/fatemeh/Downloads/fish/mot_data/ms_exp1/mots/129_2_ms_exp1.txt")
# )

# compute distances between tracks
tids1 = np.unique(tracks1[:, 0])
tids2 = np.unique(tracks2[:, 0])
all_dists = dict()
for tid1 in tids1:
    for tid2 in tids2:
        track1, track2 = pp.get_matching_frames_between_tracks(
            tracks1, tracks2, tid1, tid2
        )
        if track1.size == 0:
            continue
        c1 = normalize_curve(track1[:, 7:9])
        c2 = normalize_curve(track2[:, 7:9])
        dist = curve_distance(c1, c2)
        all_dists[(tid1, tid2)] = round(dist, 3)

# n-n matching
i2tids1 = {k: v for k, v in enumerate(tids1)}
i2tids2 = {k: v for k, v in enumerate(tids2)}
dist_mat = np.zeros((len(tids1), len(tids2)))
for i, tid1 in enumerate(tids1):
    for j, tid2 in enumerate(tids2):
        dist_mat[i, j] = all_dists.get((tid1, tid2), 10)
rows, cols = linear_sum_assignment(dist_mat)
matched_tids = [(i2tids1[r], i2tids2[c]) for r, c in zip(rows, cols)]

# 1-n matching
matched_keys = []
for tid1 in tids1:
    row_dists = {k: v for k, v in all_dists.items() if k[0] == tid1}
    if row_dists:
        match_key = min(row_dists, key=row_dists.get)
        matched_keys.append(match_key)

print(matched_tids)
print(matched_keys)
print(sorted(gt_matches.items(), key=lambda i: i[0]))

# for tid1, tid2 in gt_matches.items():
#     track1, track2 = pp.get_matching_frames_between_tracks(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(track1[::16,7],track1[::16,8],'g-*');plt.plot(track2[::16,7],track2[::16,8],'r-*');plt.show(block=False)
#     plt.figure();plt.plot(c1[::16,0],c1[::16,1],'g-*');plt.plot(c2[::16,0],c2[::16,1],'r-*');plt.show(block=False)

start, end, step = 0, 3117, 200  # 200
tid1 = 3
track1 = tracks1[tracks1[:, 0] == tid1]
tracklet1 = cut_track_into_tracklets(track1, start, end, step)
for st in range(start, end + 1, step):
    tr = tracklet1[tracklet1[:, 2] == st]
    plt.plot(tr[::16, 7], tr[::16, 8], "-*")


tracklets1 = cut_tracks_into_tracklets(tracks1, start, end, step)
tracklets2 = cut_tracks_into_tracklets(tracks2, start, end, step)

for st in range(start, end + 1, step):
    # tracklets in a specific time
    tracks1 = tracklets1[tracklets1[:, 2] == st]
    tracks2 = tracklets2[tracklets2[:, 2] == st]

    # compute distances between tracks
    tids1 = np.unique(tracks1[:, 0])
    tids2 = np.unique(tracks2[:, 0])
    all_dists = dict()
    for tid1 in tids1:
        for tid2 in tids2:
            track1, track2 = pp.get_matching_frames_between_tracks(
                tracks1, tracks2, tid1, tid2
            )
            if track1.size == 0:
                continue
            c1 = normalize_curve(track1[:, 7:9])
            c2 = normalize_curve(track2[:, 7:9])
            dist = curve_distance(c1, c2)
            all_dists[(tid1, tid2)] = round(dist, 3)

    # n-n matching
    i2tids1 = {k: v for k, v in enumerate(tids1)}
    i2tids2 = {k: v for k, v in enumerate(tids2)}
    dist_mat = np.zeros((len(tids1), len(tids2)))
    for i, tid1 in enumerate(tids1):
        for j, tid2 in enumerate(tids2):
            dist_mat[i, j] = all_dists.get((tid1, tid2), 10)
    rows, cols = linear_sum_assignment(dist_mat)
    matched_tids = [(i2tids1[r], i2tids2[c]) for r, c in zip(rows, cols)]

    print("----> ", st)
    print(matched_tids)
    print(sorted(gt_matches.items(), key=lambda i: i[0]))
