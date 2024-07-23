from pathlib import Path

import numpy as np
import pytest
from stereo_gt import (
    get_disparity_info_from_stereo_track,
    get_matched_track_ids,
    load_matched_tracks_ids,
)
from tracklet_operations import (
    add_remove_tracks,
    add_remove_tracks_by_disp_infos,
    append_tracks_with_cam_id_match_id,
    get_candidates_disparity_infos,
    get_matches_from_candidates_disparity_infos,
    match_primary_track_to_secondry_tracklets,
    remove_detects_change_track_ids,
    select_from_overlaps,
)

from tracking.data_association import (
    get_track_from_track_id,
    load_tracks_from_cvat_txt_format,
)
from tracking.postprocess import reindex_tracks, remove_short_tracks

data_path = Path(__file__).parent.parent.parent / "tests/data"
annos = load_tracks_from_cvat_txt_format(data_path / "04_07_22_G_2_rect_valid_gt.txt")
atracks = load_tracks_from_cvat_txt_format(data_path / "tracks.txt")
im_width, im_height = 2098, 1220


@pytest.mark.slow
def test_get_matched_track_ids():
    desired = np.array(load_matched_tracks_ids())

    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    matches = np.array(get_matched_track_ids(annos1, annos))
    matches = matches[matches[:, 2] < 5]
    np.testing.assert_equal(matches[:, :2], desired[:, :2])


# TODO maybe remove
def test_add_remove_tracks():
    remove_tracks = np.random.randint(10, size=(3, 2))
    remove_lengths = np.random.randint(10, size=(3, 2))
    add_tracks = np.empty(shape=(0, 2), dtype=np.int64)

    desired = remove_tracks.copy()
    inds = np.array([0])
    while remove_tracks.size != 0:
        remove_tracks, remove_lengths, add_tracks = add_remove_tracks(
            remove_tracks, remove_lengths, add_tracks, inds
        )
    np.testing.assert_equal(add_tracks, desired)
    assert remove_tracks.size == 0
    assert remove_lengths.size == 0


def test_get_disparity_info_from_stereo_track():
    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    annos2 = annos.copy()

    len_disparity = 20
    track1_id = 21
    track2_id = 16
    track1 = get_track_from_track_id(annos1, track1_id)
    track2 = get_track_from_track_id(annos2, track2_id)
    disparity_info = get_disparity_info_from_stereo_track(
        track1[: 80 + len_disparity], track2[80:150]
    )
    align_errors = disparity_info[:, 3]
    desired = np.array([1, 1, 0, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 1])
    assert disparity_info.shape[0] == len_disparity
    np.testing.assert_equal(align_errors, desired)
    np.testing.assert_equal(disparity_info[:, 0], np.repeat(track1_id, len_disparity))
    np.testing.assert_equal(disparity_info[:, 1], np.repeat(track2_id, len_disparity))
    np.testing.assert_equal(disparity_info[:, 2], np.arange(80, 100))

    # TODO: remove?


@pytest.mark.slow
def test_add_remove_tracks_by_disp_infos():
    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    annos2 = annos.copy()
    tracks1 = reindex_tracks(
        remove_short_tracks(remove_detects_change_track_ids(annos1), 10)
    )
    tracks2 = reindex_tracks(
        remove_short_tracks(remove_detects_change_track_ids(annos2), 10)
    )

    ls_tracks = np.empty(shape=(0, 13), dtype=np.int64)
    p_track = get_track_from_track_id(tracks2, 14)
    s_tracks = append_tracks_with_cam_id_match_id(tracks1, 1)
    cands = match_primary_track_to_secondry_tracklets(p_track, tracks1)
    ls_tracks, new_s_tracks = add_remove_tracks_by_disp_infos(
        cands, ls_tracks, s_tracks
    )
    track_item = new_s_tracks[(new_s_tracks[:, 0] == 72) & (new_s_tracks[:, 1] == 500)]
    assert track_item.size != 0
    assert ls_tracks.shape[0] + new_s_tracks.shape[0] == s_tracks.shape[0]


def test_get_matches_from_candidates_disparity_infos():
    annos1 = load_tracks_from_cvat_txt_format(
        data_path / "04_07_22_F_2_rect_valid_gt.txt"
    )
    annos2 = annos.copy()
    tracks1 = reindex_tracks(
        remove_short_tracks(remove_detects_change_track_ids(annos1), 10)
    )
    tracks2 = reindex_tracks(
        remove_short_tracks(remove_detects_change_track_ids(annos2), 10)
    )

    p_tracks = append_tracks_with_cam_id_match_id(tracks1, 1)
    s_tracks = append_tracks_with_cam_id_match_id(tracks2, 2)
    p_track = get_track_from_track_id(p_tracks, 20)
    cands = get_candidates_disparity_infos(p_track, s_tracks)

    sel_track_id = select_from_overlaps(4, [5, 16], cands)
    assert sel_track_id == 16
    sel_track_id = select_from_overlaps(72, [73], cands)
    assert sel_track_id == 73
    cands2 = get_matches_from_candidates_disparity_infos(cands)
    np.testing.assert_equal(np.unique(cands2[:, 1]), np.array([16, 53, 73, 83]))

    """
a = np.random.randint(10, size=(3, 2))
b = np.empty(shape=(0, 2), dtype=np.int64)
while a.size != 0:
    inds = np.array([0])
    b = np.append(b, a[inds], axis=0)
    a = np.delete(a, inds, axis=0)

# 21 <-> 16
# 21 -> 17, 72 len:(236, 332)
# 16 -> 14, 79 len: (361, 219)
# 17 <-> 14
# 72 <-> 14, 79

# 7 <-> 24
# 7  -> 5, 54, 71, 79, 84, 90 len:(80, 107, 39, 12, 58, 201)
# 24 -> 21, 63  len:(262, 313)
# 21 <-> 5, 54, 71
# 63 <-> 79, 84, 90


# tk.match_primary_track_to_secondry_tracklets(p_track1, tracks2)
# [[14, 1.0], [38, 6.0], [39, 3.0], [48, 2.0], [50, 1.0], [56, 6.0]]


p_track = da.get_track_from_track_id(annos1, 7)
s_track = da.get_track_from_track_id(annos2, 24)

fig, axs = plt.subplots(1, 2, sharex=True)
axs[0].plot(p_track[:,1], p_track[:,8],'-*',label='p_annos')
axs[0].plot(s_track[:,1], s_track[:,8],'-*',label='s_annos')

for track_id in [21, 63]:
    track = da.get_track_from_track_id(tracks2, track_id)
    axs[1].plot(track[:,1], track[:,8],'-*',label=f's_{track_id}')

for track_id in [5, 54, 71, 79, 84, 90]:
    track = da.get_track_from_track_id(tracks1, track_id)
    axs[1].plot(track[:,1], track[:,8],'-*',label=f'p_{track_id}')
axs[1].legend()
"""
