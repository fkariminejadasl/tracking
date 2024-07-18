import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import yaml

from tracking import data_association as da
from tracking.stereo_track import (
    cut_tracks_into_tracklets,
    get_stereo_parameters,
    get_t1t2s_from_g1t1s_g2t2s,
    match_gt_stereo_tracks,
    match_tracklets,
    match_tracks_with_gt,
    merge_by_mached_tids,
    merge_not_matched_tids,
    postprocess_tracks,
    rectify_tracks,
    save_images_as_video,
    save_stereo_images_with_matches_as_images,
    save_stereo_images_with_matches_as_video,
)

# from tracking import visualize
# from scipy.ndimage import uniform_filter1d # Apply a moving average to smooth data


# Parse arguments
# ===============
def process_config(config_path):
    with open(config_path, "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            print(error)


def parse_args():
    parser = argparse.ArgumentParser(description="Process a config file.")
    parser.add_argument("config_file", type=Path, help="Path to the config file")

    args = parser.parse_args()
    config_path = args.config_file
    inputs = process_config(config_path)
    for key, value in inputs.items():
        print(f"{key}: {value}")
    inputs = SimpleNamespace(**inputs)
    return inputs


# Inputs
# =======
inputs = parse_args()
save_video_file = Path(inputs.save_video_file)
vid_path1 = Path(inputs.vid_path1)
vid_path2 = Path(inputs.vid_path2)
track_file1 = Path(inputs.track_file1)
track_file2 = Path(inputs.track_file2)
if (inputs.gt_track_file1 is None) or (inputs.gt_track_file2 is None):
    gt_track_file1 = track_file1
    gt_track_file2 = track_file2
else:
    gt_track_file1 = Path(inputs.gt_track_file1)
    gt_track_file2 = Path(inputs.gt_track_file2)
if inputs.mat_file is None:
    mat_file = None
else:
    mat_file = Path(inputs.mat_file)


# parameters
# ==========
max_dist = 100

vc = cv2.VideoCapture(str(vid_path1))
end = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(round(vc.get(cv2.CAP_PROP_FPS)))  # 240
start = 0
im_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
im_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
vc.release()

print("load tracks")
tracks1 = da.load_tracks_from_mot_format(track_file1)
tracks2 = da.load_tracks_from_mot_format(track_file2)


# debuging
# ========
# visualize.save_images_with_tracks(
#                 vid_path1.parent / "images/tr",
#                 vid_path1,
#                 tracks1,
#                 0,
#                 None,
#                 1,
#             )

# postprocess tracks
# ==================
print("postprocess tracks")
tracks1 = postprocess_tracks(tracks1)
tracks2 = postprocess_tracks(tracks2)

# debuging
# ========
# visualize.save_images_with_tracks(
#                 vid_path1.parent / "images/pp",
#                 vid_path1,
#                 tracks1,
#                 0,
#                 None,
#                 1,
#             )


# rectify
# =======
# Tracks are either rectified or not. Otracks are the orignal tracks and not rectified.
otracks1 = tracks1.copy()
otracks2 = tracks2.copy()
if mat_file is not None:
    print("rectify tracks")
    (
        cameraMatrix1,
        distCoeffs1,
        R1,
        r_P1,
        cameraMatrix2,
        distCoeffs2,
        R2,
        r_P2,
    ) = get_stereo_parameters(mat_file, im_width, im_height)
    tracks1 = rectify_tracks(tracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
    tracks2 = rectify_tracks(tracks2, cameraMatrix2, distCoeffs2, R2, r_P2)

# debuging
# ========
# tid1 = 3
# track1 = tracks1[tracks1[:, 0] == tid1]
# tracklet1 = cut_track_into_tracklets(track1, start, end, step)
# for st in range(start, end + 1, step):
#     tr = tracklet1[tracklet1[:, 2] == st]
#     plt.plot(tr[::16, 7], tr[::16, 8], "-*")

# tracklet matching
# =================
print("tracklet matching")
tracklets1 = cut_tracks_into_tracklets(tracks1, start, end, step)
tracklets2 = cut_tracks_into_tracklets(tracks2, start, end, step)

frame_matches = match_tracklets(tracklets1, tracklets2, start, end, step, max_dist)
frame_matches1 = merge_by_mached_tids(frame_matches)
frame_matches2 = merge_not_matched_tids(tracks1, tracks2, frame_matches1, start, step, end)

print("matched frames: ", frame_matches2)


# debuging
# ========
# plt.figure()
# prev_disp = None
# for item in [[0, 1200, 8, 5], [1200, 2400, 8, 11], [2400, 3120, 8, 15]]:
#     track1, track2 = match_frames_within_range(tracks1, tracks2, *item)
#     disparity = track1[:,7]-track2[:,7]
#     smoothed_disp = uniform_filter1d(disparity, size=4*16)
#     if prev_disp is not None:
#         print(smoothed_disp[-1] - prev_disp)
#     prev_disp = smoothed_disp[-1]
#     plt.plot(track1[:,1], disparity, '*')
#     plt.plot(track1[:,1], smoothed_disp, 'r-')

# save results
# ============
print("save results")
# save_stereo_images_with_matches_as_images(
#     save_video_file.parent / save_video_file.stem,
#     vid_path1,
#     vid_path2,
#     otracks1,
#     otracks2,
#     frame_matches2,
#     inputs.start_frame,
#     inputs.end_frame,
#     inputs.step,
#     black=False,
# )
# save_images_as_video(
#     save_video_file,
#     save_video_file.parent / save_video_file.stem,
#     30,
#     im_width * 2,
#     im_height,
# )
save_stereo_images_with_matches_as_video(
    save_video_file,
    vid_path1,
    vid_path2,
    otracks1,
    otracks2,
    frame_matches1,
    inputs.start_frame,
    inputs.end_frame,
    inputs.step,
    inputs.fps,
    black=False,
)

"""
# evaluation
# ==========
# g1g2_frames_tids(g1->g2), g1t1_frames_tids(g1->t1), g2t2_frames_tids(g2->t2) -> (t1->t2)

# fmt: off
gt_matches = {0:5, 1:6, 2:3, 3:0, 4:2, 5:1, 6:7, 7:8, 8:4, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
gt_matches = sorted(gt_matches.items(), key=lambda x:x[0])
# fmt: on
ogtracks1 = da.load_tracks_from_mot_format(gt_track_file1)  # original, ground truth
ogtracks2 = da.load_tracks_from_mot_format(gt_track_file2)

# g1->t1, g2->t2
g1t1_tids, g1t1_frames_tids = match_tracks_with_gt(otracks1, ogtracks1)
g2t2_tids, g2t2_frames_tids = match_tracks_with_gt(otracks2, ogtracks2)
# [plot_matched_tracks_with_gt(otracks2, ogtracks2, k, v) for k, v in g1t1_tids.items()]

# g1->g2
gtracks1 = rectify_tracks(ogtracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
gtracks2 = rectify_tracks(ogtracks2, cameraMatrix2, distCoeffs2, R2, r_P2)
g1g2_tids, g1g2_frames_tids = match_gt_stereo_tracks(gtracks1, gtracks2, max_dist)

# t1-> t2
t1t2s_frames_tids = get_t1t2s_from_g1t1s_g2t2s(
    g1g2_tids, g1t1_frames_tids, g2t2_frames_tids, threshold=240
)

matched_with_gts = []
for items in frame_matches2:
    for t1t2s in t1t2s_frames_tids:
        matched = [
            [item, t1t2]
            for item in items
            for t1t2 in t1t2s
            if (item[2] == t1t2[2]) & (item[3] == t1t2[3])
        ]
        if matched:
            matched_with_gts.append(matched)


flat_frame_matches2 = [item for items in frame_matches2 for item in items]
flat_t1t2s_frames_tids = [item for items in t1t2s_frames_tids for item in items]
matched_computed = [item[0] for items in matched_with_gts for item in items]
matched_gt = [item[1] for items in matched_with_gts for item in items]
print(set(map(tuple, flat_frame_matches2)) - set(map(tuple, matched_computed)))
print(set(map(tuple, matched_computed)) - set(map(tuple, flat_frame_matches2)))
print(set(map(tuple, flat_t1t2s_frames_tids)) - set(map(tuple, matched_gt)))
print(set(map(tuple, matched_gt)) - set(map(tuple, flat_t1t2s_frames_tids)))


# evaluation
# ==========
expected = [
    [[0, 720, 0, 1], [720, 2640, 0, 9]],
    [[0, 480, 1, 3]],
    [[0, 2400, 2, 8], [2640, 3120, 2, 9]],
    [[0, 2400, 3, 6], [2400, 3120, 15, 6]],
    [[0, 1440, 4, 0], [1440, 2880, 10, 0], [2880, 3120, 19, 0]],
    [[0, 2400, 5, 7], [2400, 3120, 18, 7]],
    [[0, 2400, 6, 2], [2400, 3120, 14, 2]],
    [[0, 3120, 7, 4]],
    [[0, 1200, 8, 5], [1200, 2400, 8, 11], [2400, 3120, 8, 15]],
    [[960, 2160, 9, 10], [2160, 3120, 13, 14]],
    [[1440, 3120, 11, 12]],
    [[1920, 3120, 12, 13]],
    [[2400, 2640, 2, 17]],
    [[2400, 2640, 6, 11]],
    [[2400, 2640, 16, 8]],
    [[2400, 3120, 17, 16]],
    [[2640, 3120, 20, 17]],
    [[2880, 3120, 21, 18]],
]
assert frame_matches2 == expected
"""
print("Done")
