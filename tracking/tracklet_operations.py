import numpy as np

from tracking.data_association import (get_iou, get_track_from_track_id,
                                       get_track_inds_from_track_id)
from tracking.stereo_gt import get_disparity_info_from_stereo_track

# min_track_length: int = 10
# min_match_length = 50
# max_alignment_error = 8
# percentile = 80


def _rm_det_chang_track_id(tracks_orig: np.ndarray, frame_number: int, track_id: int):
    tracks = tracks_orig.copy()
    latest_track_id = np.unique(np.sort(tracks[:, 0]))[-1]
    ind1 = np.where((tracks[:, 1] == frame_number) & (tracks[:, 0] == track_id))[0][0]
    inds = np.where((tracks[:, 1] > frame_number) & (tracks[:, 0] == track_id))[0]
    if len(inds) != 0:
        tracks[inds, 0] = latest_track_id + 1
    tracks = np.delete(tracks, ind1, axis=0)
    return tracks


def remove_detect_change_track_id_per_frame(tracks: np.ndarray, frame_number: int):
    frame_tracks = tracks[tracks[:, 1] == frame_number].copy()
    track_ids_remove = []
    for i in range(len(frame_tracks)):
        for j in range(i + 1, len(frame_tracks)):
            item1 = frame_tracks[i]
            item2 = frame_tracks[j]
            iou = get_iou(item1[3:7], item2[3:7])
            if iou > 0:
                track_ids_remove.append(item1[0])
                track_ids_remove.append(item2[0])

    track_ids_remove = list(set(track_ids_remove))
    for track_id in track_ids_remove:
        tracks = _rm_det_chang_track_id(tracks, frame_number, track_id)
    return tracks


def remove_detects_change_track_ids(tracks: np.ndarray):
    frame_numbers = np.unique(np.sort(tracks[:, 1]))
    for frame_number in frame_numbers:
        tracks = remove_detect_change_track_id_per_frame(tracks, frame_number)
    return tracks


def remove_short_tracks(tracks: np.ndarray, min_track_length: int = 10):
    track_ids = np.unique(np.sort(tracks[:, 0]))
    for track_id in track_ids:
        inds = np.where(tracks[:, 0] == track_id)[0]
        if len(inds) < min_track_length:
            tracks = np.delete(tracks, inds, axis=0)
    return tracks


def arrange_track_ids(tracks: np.ndarray):
    new_tracks = tracks.copy()
    track_ids = np.unique(np.sort(tracks[:, 0]))
    old_to_new_ids = {
        track_id: new_track_id for new_track_id, track_id in enumerate(track_ids)
    }
    for track_id in track_ids:
        new_tracks[tracks[:, 0] == track_id, 0] = old_to_new_ids[track_id]
    return new_tracks


# TODO: remove?
def compute_tracks_length_same_size(tracks: np.ndarray) -> np.ndarray:
    lengths = tracks[:, :3].copy()
    tracks_ids = np.unique(tracks[:, 0])
    for track_id in tracks_ids:
        track = get_track_from_track_id(tracks, track_id)
        lengths[lengths[:, 0] == track_id, 2] = len(track)
    return lengths


# TODO: remove?
def add_remove_tracks(
    remove_tracks: np.ndarray,
    remove_lengths: np.ndarray,
    add_tracks: np.ndarray,
    inds: np.ndarray,
):
    add_tracks = np.append(add_tracks, remove_tracks[inds], axis=0)
    remove_tracks = np.delete(remove_tracks, inds, axis=0)
    remove_lengths = np.delete(remove_lengths, inds, axis=0)
    return remove_tracks, remove_lengths, add_tracks


def get_track_length(tracks, track_id):
    track = get_track_from_track_id(tracks, track_id)
    track_length = len(track)
    return track_length


def _compute_tracks_lengths(tracks, cam_id) -> list:
    # each track_length: track_id, cam_id, track_length
    tracks_lengths = []
    tracks_ids = np.unique(tracks[:, 0])
    for track_id in tracks_ids:
        track_length = get_track_length(tracks, track_id)
        tracks_lengths.append([track_id, cam_id, track_length])
    return tracks_lengths


def append_tracks_with_cam_id_match_id(tracks: np.ndarray, cam_id: int):
    array_to_append = np.repeat([[cam_id, -1]], tracks.shape[0], axis=0)
    return np.concatenate((tracks, array_to_append), axis=1)


# TODO: track shape is changed: appended with (camera_id, match_id)
def compute_two_tracks_lengths_sorted_descending(
    tracks1: np.ndarray, tracks2: np.ndarray
) -> np.ndarray:
    # each track_length: track_id, cam_id, track_length
    assert tracks1.shape[1] == 13 and tracks2.shape[1] == 13
    tracks_lengths = _compute_tracks_lengths(tracks1, tracks1[0, -2])
    tracks_lengths.extend(_compute_tracks_lengths(tracks2, tracks2[0, -2]))
    tracks_lengths = np.array(tracks_lengths).astype(np.int64)
    inds = np.argsort(tracks_lengths[:, 2], axis=0)[::-1]
    tracks_lengths = tracks_lengths[inds]
    return tracks_lengths


def define_primary_secondary_tracks(
    cam_id, tracks1, long_tracks1, tracks2, long_tracks2
):
    if cam_id == tracks1[0, -2]:
        return tracks1, long_tracks1, tracks2, long_tracks2
    else:
        return tracks2, long_tracks2, tracks1, long_tracks1


# TODO maybe add new ids here
def add_remove_tracks_by_track_ids(tracks_ids, add_tracks, remove_tracks):
    for track_id in tracks_ids:
        track = get_track_from_track_id(remove_tracks, track_id)
        inds = get_track_inds_from_track_id(remove_tracks, track_id)
        add_tracks = np.append(add_tracks, track, axis=0)
        remove_tracks = np.delete(remove_tracks, inds, axis=0)
    return add_tracks, remove_tracks


def get_a_matched_disparity_info(
    track1, track2, percentile=80, max_alignment_error=8, min_match_length=10
):
    disparity_info = np.array(get_disparity_info_from_stereo_track(track1, track2))
    if disparity_info.size > 0:
        align_error = np.percentile(disparity_info[:, 3], percentile)
        align_error = float(f"{align_error:.2f}")
        if align_error < max_alignment_error:
            sel_disparity_info = disparity_info[disparity_info[:, 3] < align_error]
            match_length = sel_disparity_info.shape[0]
            if match_length > min_match_length:
                length_and_error_repeated = np.broadcast_to(
                    [match_length, align_error], (sel_disparity_info.shape[0], 2)
                )
                return np.concatenate(
                    (sel_disparity_info, length_and_error_repeated), axis=1
                )
    return np.empty(shape=(0), dtype=np.int64)


def get_matches_from_candidates_disparity_infos(
    candidates_disparity_infos, min_match_length=50
):
    frame_numbers = np.unique(candidates_disparity_infos[:, 2])
    matched_disparity_infos = []
    for frame_number in frame_numbers:
        candidates = candidates_disparity_infos[
            candidates_disparity_infos[:, 2] == frame_number
        ]
        max_length = max(candidates[:, 5])
        # min_align_error = min(candidates[:, 6])
        matched_disparity_info = candidates[candidates[:, 5] == max_length][0]
        matched_disparity_infos.append(matched_disparity_info)
    matched_disparity_infos = np.array(matched_disparity_infos)
    tracks_ids = np.unique(matched_disparity_infos[:, 1])
    # TODO hack to remove non-overlapping short tracks
    matched_s_tracks_ids = []
    for track_id in tracks_ids:
        len_matched_track = len(
            matched_disparity_infos[matched_disparity_infos[:, 1] == track_id]
        )
        if len_matched_track > min_match_length:
            matched_s_tracks_ids.append(track_id)
    return matched_s_tracks_ids


def match_primary_track_to_secondry_tracklets(p_track, s_tracks):
    assert np.unique(p_track[:, 0]).size == 1
    s_tracks_ids = np.unique(s_tracks[:, 0])
    candidates_disparity_infos = []
    for s_track_id in s_tracks_ids:
        s_track = get_track_from_track_id(s_tracks, s_track_id)
        matched_disparity_info = get_a_matched_disparity_info(p_track, s_track)
        if matched_disparity_info.size != 0:
            candidates_disparity_infos.extend(matched_disparity_info)
    candidates_disparity_infos = np.array(candidates_disparity_infos)
    matched_s_tracks_ids = get_matches_from_candidates_disparity_infos(
        candidates_disparity_infos
    )
    return matched_s_tracks_ids
