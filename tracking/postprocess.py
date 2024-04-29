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


def correct_outliers(track, max_disp=10):
    """
    example for gtrack1 7:8 frame 1400 has issues (129_1)
    track[np.arange(1396,1404+1),7] = np.interp(np.arange(1396,1404+1), [1396,1404], [track[1396,7], track[1404,7]])
    track[np.arange(1396,1404+1),8] = np.interp(np.arange(1396,1404+1), [1396,1404], [track[1396,8], track[1404,8]])
    """
    argmax_diff_disp = np.argmax(abs(np.diff(np.linalg.norm(track[:, 7:9], axis=1))))
    max_diff_disp = np.max(abs(np.diff(np.linalg.norm(track[:, 7:9], axis=1))))
    if max_diff_disp > max_disp:
        ind1 = argmax_diff_disp - 4
        ind2 = argmax_diff_disp + 4
        track[np.arange(ind1, ind2 + 1), 7] = np.interp(
            np.arange(ind1, ind2 + 1), [ind1, ind2], [track[ind1, 7], track[ind2, 7]]
        )
        track[np.arange(ind1, ind2 + 1), 8] = np.interp(
            np.arange(ind1, ind2 + 1), [ind1, ind2], [track[ind1, 8], track[ind2, 8]]
        )
    return track


def remove_static_tracks(tracks, window_size=16, move_threshold=10):
    """
    Removes static tracks based on the moving average position over a specified window size.

    Parameters:
    - tracks (np.ndarray): Array containing track data with columns as follows:
        [track_id, frame_number, det_id, x_tl, y_tl, x_br, y_br, x_cen, y_cen, width, height]
    - window_size (int): The size of the rolling window for computing the moving average.
    - move_threshold (int): The minimum distance the moving average position of a track must move.

    Returns:
    - np.ndarray: Array with static tracks removed.
    """
    track_ids = np.unique(tracks[:, 0])
    non_static_indices = []

    for track_id in track_ids:
        track = tracks[tracks[:, 0] == track_id]  # copy
        x_centers = track[:, 7]
        y_centers = track[:, 8]

        ma_x = np.convolve(x_centers, np.ones(window_size) / window_size, mode="valid")
        ma_y = np.convolve(y_centers, np.ones(window_size) / window_size, mode="valid")
        ma_x = x_centers[::window_size]  # stride
        ma_y = y_centers[::window_size]

        distances = np.sqrt((ma_x[1:] - ma_x[0]) ** 2 + (ma_y[1:] - ma_y[0]) ** 2)
        if np.all(distances < move_threshold):
            print(track_id)
        else:
            non_static_indices.append(np.where(tracks[:, 0] == track_id)[0])

    return tracks[np.concatenate(non_static_indices)]


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
        bbox = np.int64(np.round(bbox, decimals=0))
        cent_wh = np.int64(np.round(cent_wh, decimals=0))
        detection = [track_id, frame_number, -1, *bbox, *cent_wh, -1]  # TODO
        detections.append(detection)
    new_track = np.concatenate((track, detections), axis=0)
    new_track = new_track[new_track[:, 1].argsort()]
    return new_track


def interpolate_track_when_missing_frames(track):
    frame_numbers = np.unique(track[:, 1])
    all_frame_numbers = np.arange(frame_numbers[0], frame_numbers[-1] + 1)
    missing_frames = np.array(sorted(set(all_frame_numbers).difference(frame_numbers)))
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
    new_tracks = np.empty(shape=(0, tracks.shape[1]), dtype=np.int64)
    tracks_ids = np.unique(tracks[:, 0])
    for track_id in tracks_ids:
        track = tracks[tracks[:, 0] == track_id]  # copy
        new_track = interpolate_track_when_missing_frames(track)
        new_tracks = np.concatenate((new_tracks, new_track), axis=0)
    return new_tracks


def tid_from_xyf(tracks, xcent, ycent, frame, thrs=1):
    """
    get track id from detection center x, y and frame number
    """
    mask = (
        (abs(tracks[:, 7] - xcent) < thrs)
        & (abs(tracks[:, 8] - ycent) < thrs)
        & (tracks[:, 1] == frame)
    )
    if len(tracks[mask]) == 0:
        return
    track_id = tracks[mask][0, 0]
    print(tracks[mask])
    return track_id


def merge_two_tracks_by_speed(track1, track2, thrs=6):
    """Merge two tracks
    Due to (partial) occlusions, there is id switchs.
    E.g. groudtruth track 1 consists of track 6, 70
    """
    # check with track follows other track
    if track2[-1, 1] < track1[0, 1]:
        tmp = track2.copy()
        track2 = track1.copy()
        track1 = tmp
    # normalized displacement (not a real speed)
    disp = np.linalg.norm(track2[0, 7:9] - track1[-1, 7:9]) / (
        track2[0, 1] - track1[-1, 1]
    )
    if disp < thrs:
        tid = track1[0, 0]
        track = np.concatenate((track1, track2), axis=0)  # copy
        track[:, 0] = tid
    return track


def plot_2d_tracks(tracks):
    track_ids = np.unique(tracks[:, 0])
    plt.figure()
    for tid in track_ids:
        track = tracks[tracks[:, 0] == tid]
        plt.plot(track[:, 7], track[:, 8], "-*", label=str(tid))
    plt.gca().invert_yaxis()
    plt.legend()


# fmt: off
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# fmt: on
gtracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_1.txt")
)
gtracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/mots/129_2.txt")
)
tracks1 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/ms_exp1/mots/129_1_ms_exp1.txt")
)
tracks2 = da.load_tracks_from_mot_format(
    Path("/home/fatemeh/Downloads/fish/mot_data/ms_exp1/mots/129_2_ms_exp1.txt")
)

# gtrack1: 0->0,72,74 (16.76 0-72); 1->6,70 (6.1); 2->1; 3->7; 4->9; 5->4,51,81 (10.8 4-51); 6->3,71; 7->5,75 (8.3);
# 8->2; 9->36; 10->55; 11->61; 12->67; 13->82; 14->84
# track 2(larger disp 6.2,10.2)
# [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 36, 51, 55, 61, 67, 70, 71, 72, 74, 75, 81, 82, 84]
gtrack1 = gtracks1[(gtracks1[:, 0] == 1)]
for det in gtrack1[::16]:
    tid_from_xyf(tracks1, det[7], det[8], det[1], thrs=5)
track1 = tracks1[(tracks1[:, 0] == 6)]
track2 = tracks1[(tracks1[:, 0] == 70)]


plt.figure()
plt.plot(gtrack1[:, 7], gtrack1[:, 8], "*-g")
plt.plot(track1[:, 7], track1[:, 8], "o--")
plt.plot(track2[:, 7], track2[:, 8], "o--")

track = merge_two_tracks_by_speed(track1, track2, thrs=7)
new_track = interpolate_track_when_missing_frames(track)

disp1 = abs(np.diff(np.linalg.norm(track1[:, 7:9], axis=1))) / np.diff(track1[:, 1])
disp2 = abs(np.diff(np.linalg.norm(track2[:, 7:9], axis=1))) / np.diff(track2[:, 1])
plt.figure()
plt.plot(track1[1:, 1], disp1, "*-")
plt.plot(track2[1:, 1], disp2, "*-")

track_lens1 = get_tracks_lengths(tracks1)
sorted(track_lens1.items(), key=lambda x: x[1])


# remove static tracks, remove short tracks, split and merge tracks, reindexing, interpolate tracks, correct outliers (?)
tracks = tracks1.copy()
tracks = remove_static_tracks(tracks, window_size=16, move_threshold=10)
tracks = remove_short_tracks(tracks, min_track_length=16)
track = merge_two_tracks_by_speed(track1, track2, thrs=7)
s_tracks = tracks[np.isin(tracks[:, 0], [0, 1, 2, 3, 4, 5, 6, 7, 9])]
plot_2d_tracks(s_tracks)
tracks = reindex_tracks(tracks)
tracks = interpolate_tracks_when_missing_frames(tracks)
plot_2d_tracks(tracks)


# fmt: off
# TODO: remove
# 7->48, 1-> 40, 8,0-> 30 rest <8 (6 near 8,0 but different)
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# gt_matches = {3:0, 5:0, 4:0, 2:0, 8:0, 0:0, 1:0, 6:0, 7:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}

track1, track2 = get_matching_frames_between_tracks(tracks1, tracks2, 7, 8)
track1 = correct_outliers(track1)

plt.figure();plt.plot(track1[::16,7],track1[::16,8],'g-*');plt.plot(track2[::16,7],track2[::16,8],'r-*');plt.show(block=False)
# plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{id1}:{id2}");plt.show(block=False)
# np.mean(np.linalg.norm(np.diff(track1[:,7:9]-track2[:,7:9],axis=0),axis=1))

# for tid1, tid2 in gt_matches.items():
#     track1, track2 = get_matching_frames_between_tracks(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(np.diff(np.linalg.norm(track1[:,7:9], axis=1)),'-*');plt.plot(np.diff(np.linalg.norm(track2[:,7:9], axis=1)),'-*');plt.title(f"{tid1}:{tid2}")

# TODO: remove
# fmt: on
