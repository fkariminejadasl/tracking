import numpy as np
from pathlib import Path

from tracking.data_association import (
    get_a_track_from_track_id,
)


def get_disparity_info_from_stereo_track(track1: np.ndarray, track2:np.ndarray) -> np.ndarray:
    min_frame_number = min(min(track1[:,1]), min(track2[:,1]))
    max_frame_number = max(max(track1[:,1]), max(track2[:,1]))
    disparity_info = []
    for frame_number in range(min_frame_number, max_frame_number+1):
        det1 = track1[track1[:,1]==frame_number]
        det2 = track2[track2[:,1]==frame_number]
        if det1.size>0 and det2.size>0:
            det1 = det1[0]
            det2 = det2[0]
            align_error = abs(det1[8] - det2[8])
            disparity = abs(det1[7] - det2[7])
            disparity_info.append([det1[0], det2[0], frame_number, align_error, disparity])
    return disparity_info



def get_disparity_info_from_stereo_tracks(tracks1:np.ndarray, tracks2:np.ndarray, matches: np.ndarray):
    disparity_infos = []
    tracks1_ids = np.unique(tracks1[:,0])
    for track1_id in tracks1_ids:
        ind = np.where(matches[:,0]==track1_id)[0][0]
        track2_id = matches[ind, 1]
        if track2_id != -1:
            track1 = get_a_track_from_track_id(tracks1, track1_id)
            track2 = get_a_track_from_track_id(tracks2, track2_id)
            disparity_info = get_disparity_info_from_stereo_track(track1, track2)
            disparity_infos.extend(disparity_info)
    return disparity_infos

def get_mean_alignment_error(track1, track2):
    disparity_info = np.array(get_disparity_info_from_stereo_track(track1, track2))
    if disparity_info.size > 0:
        return np.mean(disparity_info[:, 3])
    return None


def get_matched_track_ids(tracks1, tracks2):
    tracks1_ids = np.unique(tracks1[:, 0])
    tracks2_ids = np.unique(tracks2[:, 0])
    matches = []
    for track1_id in tracks1_ids:
        align_errors = []
        for track2_id in tracks2_ids:
            track1 = get_a_track_from_track_id(tracks1, track1_id)
            track2 = get_a_track_from_track_id(tracks2, track2_id)
            align_error = get_mean_alignment_error(track1, track2)
            if align_error:
                align_errors.append(
                    [track2_id, get_mean_alignment_error(track1, track2)]
                )
        align_errors = np.array(align_errors)
        track2_id, align_error = align_errors[
            align_errors[:, 1] == min(align_errors[:, 1])
        ][0]
        matches.append([track1_id, int(track2_id), float(f"{align_error:.2f}")])
    return matches


def load_matched_tracks_ids():
    stereo_matches = []
    with open("/home/fatemeh/dev/tracking/tests/data/stereo_matches.txt", "r") as rfile:
        rfile.readline()
        for row in rfile:
            items = row.split("\n")[0].split(",")
            track1_id = int(items[0])
            track2_id = int(items[1])
            stereo_matches.append([track1_id, track2_id])
    return stereo_matches


def save_matched_tracks_ids(matches):
    with open("/home/fatemeh/dev/tracking/tests/data/stereo_matches.txt", "w") as rfile:
        rfile.write("track1_id, track2_id, mean align_error")
        for item in matches:
            if item[2] < 5:
                rfile.write("\n")
                rfile.write(f"{item[0]},{item[1]},{item[2]}")



