from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import linear_sum_assignment

from tracking import data_association as da


def find_matching_frames_for_matched_track(tracks1, tracks2, track_id1, track_id2):
    # get the matched detections based on the commond frame numbers
    track1 = tracks1[tracks1[:, 0] == track_id1]
    track2 = tracks2[tracks2[:, 0] == track_id2]
    common_frames = np.intersect1d(track1[:, 1], track2[:, 1])
    track1 = track1[np.isin(track1[:, 1], common_frames)]
    track2 = track2[np.isin(track2[:, 1], common_frames)]
    return track1, track2


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
        track1, track2 = find_matching_frames_for_matched_track(
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
#     track1, track2 = find_matching_frames_for_matched_track(tracks1, tracks2, tid1, tid2)
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
            track1, track2 = find_matching_frames_for_matched_track(
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

# ======================================
# Move to tracklet_operations or track postprocessing. Some of the functions are reimplemented accidentally.
def correct_outliers(track1, track2, max_disp=10):
    """
    example for track 7:8 frame 1400 has issues
    track1[np.arange(1396,1404+1),7] = np.interp(np.arange(1396,1404+1), [1396,1404], [track1[1396,7], track1[1404,7]])
    track1[np.arange(1396,1404+1),8] = np.interp(np.arange(1396,1404+1), [1396,1404], [track1[1396,8], track1[1404,8]])
    """
    argmax_diff_disp = np.argmax(abs(np.diff(np.linalg.norm(track1[:, 7:9], axis=1))))
    max_diff_disp = np.max(abs(np.diff(np.linalg.norm(track1[:, 7:9], axis=1))))
    if max_diff_disp > max_disp:
        ind1 = argmax_diff_disp - 4
        ind2 = argmax_diff_disp + 4
        track1[np.arange(ind1, ind2 + 1), 7] = np.interp(
            np.arange(ind1, ind2 + 1), [ind1, ind2], [track1[ind1, 7], track1[ind2, 7]]
        )
        track1[np.arange(ind1, ind2 + 1), 8] = np.interp(
            np.arange(ind1, ind2 + 1), [ind1, ind2], [track1[ind1, 8], track1[ind2, 8]]
        )
    return track1, track2


# TODO: rem. these functions are already implemented tracklet_operations


def remove_short_tracks(tracks, min_track_length):
    # Group by track_id (first column of the array) and filter by length
    unique_track_ids = np.unique(tracks[:, 0])
    new_tracks = []
    for track_id in unique_track_ids:
        track = tracks[tracks[:, 0] == track_id]
        if len(track) > min_track_length:
            new_tracks.extend(track)
    return np.array(new_tracks)


def reindex_tracks(tracks):
    # Reindex all tracks to have consecutive track_ids starting from 0
    unique_track_ids = np.unique(tracks[:, 0])
    new_id = 0
    new_tracks = []
    for old_id in unique_track_ids:
        track = tracks[tracks[:, 0] == old_id]
        # change track id
        track[:, 0] = new_id
        new_tracks.extend(track)
        new_id += 1
    return np.array(new_tracks)


def get_track_lengths(tracks):
    tids = np.unique(tracks[:, 0])
    track_lens = dict()
    for tid in tids:
        track = tracks1[tracks1[:, 0] == tid]
        track_lens[tid] = len(track)
    return track_lens


track_lens1 = get_track_lengths(tracks1)  # tk.get_tracks_lengths
sorted(track_lens1.items(), key=lambda x: x[1])

tracks1_lt = remove_short_tracks(tracks1, 10)  # tk.remove_short_tracks
track_lt_lens1 = get_track_lengths(tracks1_lt)
tracks1_lt_ri = reindex_tracks(tracks1_lt)  # tk.arrange_track_ids

track1 = tracks1[tracks1[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")
track1 = tracks1_lt[tracks1_lt[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")
track1 = tracks1_lt_ri[tracks1_lt_ri[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")


# fmt: off
# TODO: remove
# 7->48, 1-> 40, 8,0-> 30 rest <8 (6 near 8,0 but different)
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# gt_matches = {3:0, 5:0, 4:0, 2:0, 8:0, 0:0, 1:0, 6:0, 7:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}

track1, track2 = find_matching_frames_for_matched_track(tracks1, tracks2, 7, 8)
track1, track2 = correct_outliers(track1, track2)

plt.figure();plt.plot(track1[::16,7],track1[::16,8],'g-*');plt.plot(track2[::16,7],track2[::16,8],'r-*');plt.show(block=False)
# plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{id1}:{id2}");plt.show(block=False)
# np.mean(np.linalg.norm(np.diff(track1[:,7:9]-track2[:,7:9],axis=0),axis=1))

# for tid1, tid2 in gt_matches.items():
#     track1, track2 = find_matching_frames_for_matched_track(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{tid1}:{tid2}")

# TODO: remove
# fmt: on
