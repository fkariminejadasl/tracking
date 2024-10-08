import numpy as np

from tracking.data_association import get_track_from_track_id


def get_disparity_info_from_stereo_track(
    track1: np.ndarray, track2: np.ndarray
) -> np.ndarray:
    # disparity_info: track_id1, track_id2, frame_number, align_error, disparity
    frame_numbers1 = np.unique(track1[:, 1])
    frame_numbers2 = np.unique(track2[:, 1])
    common_frame_numbers = set(frame_numbers1).intersection(set(frame_numbers2))
    disparity_info = []
    for frame_number in common_frame_numbers:
        det1 = track1[track1[:, 1] == frame_number]
        det2 = track2[track2[:, 1] == frame_number]
        if det1.size > 0 and det2.size > 0:
            det1 = det1[0]
            det2 = det2[0]
            align_error = abs(det1[8] - det2[8])
            disparity = abs(det1[7] - det2[7])
            disparity_info.append(
                [det1[0], det2[0], frame_number, align_error, disparity]
            )
    disparity_info = np.array(disparity_info).astype(np.int64)
    return disparity_info


# TODO remove?
def get_disparity_info_from_stereo_tracks(
    tracks1: np.ndarray, tracks2: np.ndarray, matches: np.ndarray
):
    disparity_infos = np.empty(shape=(0, 5))
    tracks1_ids = np.unique(tracks1[:, 0])
    for track1_id in tracks1_ids:
        ind = np.where(matches[:, 0] == track1_id)[0][0]
        track2_id = matches[ind, 1]
        if track2_id != -1:
            track1 = get_track_from_track_id(tracks1, track1_id)
            track2 = get_track_from_track_id(tracks2, track2_id)
            disparity_info = get_disparity_info_from_stereo_track(track1, track2)
            disparity_infos = np.append(disparity_infos, disparity_info, axis=0)
    return disparity_infos


def get_mean_alignment_error(track1, track2):
    disparity_info = get_disparity_info_from_stereo_track(track1, track2)
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
            track1 = get_track_from_track_id(tracks1, track1_id)
            track2 = get_track_from_track_id(tracks2, track2_id)
            align_error = get_mean_alignment_error(track1, track2)
            if align_error:
                align_errors.append([track2_id, align_error])
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


# Old visualziation
import matplotlib.pylab as plt


def plot_disparity_info(disparity_info: np.ndarray, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2)
    assert len(axs) >= 2
    label = f"{disparity_info[0,0]},{disparity_info[0,1]}"
    axs[0].plot(disparity_info[:, 2], disparity_info[:, 3], label=label)
    axs[1].plot(disparity_info[:, 2], disparity_info[:, 4], label=label)
    axs[0].legend()
    axs[1].legend()


def plot_disparity_infos(tracks1, tracks2, matches):
    for i, (track1_id, track2_id) in enumerate(matches):
        track1 = get_track_from_track_id(tracks1, track1_id)
        track2 = get_track_from_track_id(tracks2, track2_id)
        disparity_info = get_disparity_info_from_stereo_track(track1, track2)
        print(track1_id, track2_id, len(disparity_info))
        if i % 10 == 0:
            fig, axs = plt.subplots(1, 2)
        plot_disparity_info(disparity_info, axs)
