from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pylab as plt

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

tids1 = sorted(gt_matches.keys())
tids2 = sorted(gt_matches.values())
all_dists = dict()
row_dists = dict()
matched_keys = list()
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
        row_dists[(tid1, tid2)] = round(dist, 3)
    match_key = min(row_dists, key=row_dists.get)
    matched_keys.append(match_key)
    row_dists = dict()

print(matched_keys)
print(sorted(gt_matches.items(), key=lambda i: i[0]))

# for tid1, tid2 in gt_matches.items():
#     track1, track2 = find_matching_frames_for_matched_track(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(track1[::16,7],track1[::16,8],'g-*');plt.plot(track2[::16,7],track2[::16,8],'r-*');plt.show(block=False)
#     plt.figure();plt.plot(c1[::16,0],c1[::16,1],'g-*');plt.plot(c2[::16,0],c2[::16,1],'r-*');plt.show(block=False)

start, end, step = 0, 3117, 1000  # 200
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

    # same code for all tracks
    all_dists = dict()
    row_dists = dict()
    matched_keys = list()
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
            row_dists[(tid1, tid2)] = round(dist, 3)
        if len(row_dists) == 0:
            continue
        match_key = min(row_dists, key=row_dists.get)
        matched_keys.append(match_key)
        row_dists = dict()

    print(matched_keys)
    print(sorted(gt_matches.items(), key=lambda i: i[0]))
    print("----> ", st)
