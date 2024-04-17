from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import linear_sum_assignment

from tracking import data_association as da


def get_matching_frames_between_tracks(tracks1, tracks2, track_id1, track_id2):
    # get the matched detections based on the common frame numbers
    track1 = tracks1[tracks1[:, 0] == track_id1]
    track2 = tracks2[tracks2[:, 0] == track_id2]
    common_frames = np.intersect1d(track1[:, 1], track2[:, 1])
    track1 = track1[np.isin(track1[:, 1], common_frames)]
    track2 = track2[np.isin(track2[:, 1], common_frames)]
    return track1, track2


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


def remove_short_tracks(tracks: np.ndarray, min_track_length: int = 10):
    track_ids = np.unique(tracks[:, 0])
    new_tracks = []
    for track_id in track_ids:
        track = tracks[tracks[:, 0] == track_id]
        if len(track) >= min_track_length:
            new_tracks.extend(track)
    return np.array(new_tracks)


def reindex_tracks(tracks: np.ndarray):
    """Reindex all tracks to have consecutive track_ids starting from 0"""
    new_tracks = tracks.copy()
    track_ids = np.unique(tracks[:, 0])
    old_to_new_ids = {
        track_id: new_track_id for new_track_id, track_id in enumerate(track_ids)
    }
    for track_id in track_ids:
        new_tracks[tracks[:, 0] == track_id, 0] = old_to_new_ids[track_id]
    return new_tracks


def get_tracks_lengths(tracks):
    tids = np.unique(tracks[:, 0])
    track_lens = dict()
    for tid in tids:
        track = tracks[tracks[:, 0] == tid]  # copy
        track_lens[tid] = len(track)
    return track_lens


def interpolate_two_bboxes(start_bbox, end_bbox, start_frame, end_frame):
    # bbox: (x_topleft, y_topleft, x_bottomright, y_bottomright)
    frames = np.arange(start_frame + 1, end_frame)
    given_frames = [start_frame, end_frame]
    xs_tl = np.interp(frames, given_frames, [start_bbox[0], end_bbox[0]])
    ys_tl = np.interp(frames, given_frames, [start_bbox[1], end_bbox[1]])
    xs_br = np.interp(frames, given_frames, [start_bbox[2], end_bbox[2]])
    ys_br = np.interp(frames, given_frames, [start_bbox[3], end_bbox[3]])
    return list(zip(xs_tl, ys_tl, xs_br, ys_br))


def tl_br_from_cen_wh(center_x, center_y, bbox_w, bbox_h):
    return (
        int(round(center_x - bbox_w / 2)),
        int(round(center_y - bbox_h / 2)),
        int(round(center_x + bbox_w / 2)),
        int(round(center_y + bbox_h / 2)),
    )


def cen_wh_from_tl_br(tl_x, tl_y, br_x, br_y):
    width = int(round(br_x - tl_x))
    height = int(round(br_y - tl_y))
    center_x = int(round(width / 2 + tl_x))
    center_y = int(round(height / 2 + tl_y))
    return center_x, center_y, width, height


def get_start_ends_missing_frames(missing_frames):
    diffs = np.diff(missing_frames)
    inds = np.where(diffs != 1)[0]

    diffs1 = np.hstack((missing_frames[0], missing_frames[inds + 1]))
    diffs2 = np.hstack((missing_frames[inds], missing_frames[-1]))

    start_ends = []
    for start_frame, end_frame in zip(diffs1, diffs2):
        start_ends.append((start_frame - 1, end_frame + 1))
    return start_ends


def make_new_detections_for_track(track, bboxes, start_frame, end_frame):
    track_id = track[0, 0]
    frame_numbers = np.arange(start_frame + 1, end_frame)
    detections = []
    for frame_number, bbox in zip(frame_numbers, bboxes):
        cent_wh = cen_wh_from_tl_br(*bbox)
        detection = [frame_number, track_id, -1, *bbox, *cent_wh]  # TODO
        detections.append(detection)
    new_track = np.concatenate((track, detections), axis=0)
    return new_track


def interpolate_track_when_missing_frames(track):
    frame_numbers = np.unique(track[:, 1])
    all_frame_numbers = np.arange(frame_numbers[0], frame_numbers[-1] + 1)
    missing_frames = set(all_frame_numbers).difference(set(frame_numbers))
    if len(missing_frames) == 0:
        return track
    new_track = track.copy()
    start_ends = get_start_ends_missing_frames(missing_frames)
    for start_frame, end_frame in start_ends:
        start_bbox = track[track[:, 1] == start_frame, 3:7][0]
        end_bbox = track[track[:, 1] == end_frame, 3:7][0]
        bboxes = interpolate_two_bboxes(start_bbox, end_bbox, start_frame, end_frame)
        new_track = make_new_detections_for_track(
            new_track, bboxes, start_frame, end_frame
        )
    return new_track


def interpolate_tracks_when_missing_frames(tracks):
    new_tracks = np.empty(shape=(0, 11), dtype=np.int64)
    tracks_ids = np.unique(tracks[:, 0])
    for track_id in tracks_ids:
        track = tracks[tracks[:, 0] == track_id]  # copy
        new_track = interpolate_track_when_missing_frames(track)
        new_tracks = np.concatenate((new_tracks, new_track), axis=0)
    return new_tracks


# remove static tracks, remove short tracks, interpolate tracks, correct outliers, reindexing
# remove_short_tracks, interpolate_tracks_when_missing_frames, correct_outliers, reindex_tracks

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

track_lens1 = get_tracks_lengths(tracks1)
sorted(track_lens1.items(), key=lambda x: x[1])

tracks1_lt = remove_short_tracks(tracks1, 10)
track_lt_lens1 = get_tracks_lengths(tracks1_lt)
tracks1_lt_ri = reindex_tracks(tracks1_lt)

track1 = tracks1[tracks1[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")
track1 = tracks1_lt[tracks1_lt[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")
track1 = tracks1_lt_ri[tracks1_lt_ri[:, 0] == 7]
plt.figure()
plt.plot(track1[::16, 7], track1[::16, 8], "g-*")


tracks = tracks1_lt.copy()


def plot_2d_tracks(tracks):
    track_ids = np.unique(tracks[:, 0])
    plt.figure()
    for tid in track_ids:
        track = tracks[tracks[:, 0] == tid]
        plt.plot(track[:, 7], track[:, 8], "-*")


# fmt: off
# TODO: remove
# 7->48, 1-> 40, 8,0-> 30 rest <8 (6 near 8,0 but different)
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# gt_matches = {3:0, 5:0, 4:0, 2:0, 8:0, 0:0, 1:0, 6:0, 7:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}

track1, track2 = get_matching_frames_between_tracks(tracks1, tracks2, 7, 8)
track1, track2 = correct_outliers(track1, track2)

plt.figure();plt.plot(track1[::16,7],track1[::16,8],'g-*');plt.plot(track2[::16,7],track2[::16,8],'r-*');plt.show(block=False)
# plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{id1}:{id2}");plt.show(block=False)
# np.mean(np.linalg.norm(np.diff(track1[:,7:9]-track2[:,7:9],axis=0),axis=1))

# for tid1, tid2 in gt_matches.items():
#     track1, track2 = get_matching_frames_between_tracks(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{tid1}:{tid2}")

# TODO: remove
# fmt: on
