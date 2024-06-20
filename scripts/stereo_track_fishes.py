import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from matplotlib import pylab as plt
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking import data_association as da
from tracking import postprocess as pp

"""
TODO id dictionary after reindex 
TODO match_tracks_with_gt is very basic. Since due to id switch, part of id
can be matched to gt. The same for plot_matched_tracks_with_gt.
e.g. in 129_1, 2 in gt 6 wrong (crossing); 14 in 5
e.g. in 129_2, 2 in gt 2 wrong; 6, 8, 9 in 5 (part 8, 9 correct)
"""


# Matching
# ========
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


# 3D
def cen_wh_from_tl_br(tl_x, tl_y, br_x, br_y):
    width = int(round(br_x - tl_x))
    height = int(round(br_y - tl_y))
    center_x = int(round(width / 2 + tl_x))
    center_y = int(round(height / 2 + tl_y))
    return center_x, center_y, width, height


def rectify_detections(dets, cameraMatrix, distCoeffs, R, r_P):
    r_dets = dets.copy()
    tl_pts = dets[:, 3:5].astype(np.float32)  # top left points
    br_pts = dets[:, 5:7].astype(np.float32)  # bottom right points

    # rectify points
    rtl_pts = cv2.undistortPoints(tl_pts, cameraMatrix, distCoeffs, R=R, P=r_P).squeeze(
        axis=1
    )
    rbr_pts = cv2.undistortPoints(br_pts, cameraMatrix, distCoeffs, R=R, P=r_P).squeeze(
        axis=1
    )
    for r_det, tl, br in zip(r_dets, rtl_pts, rbr_pts):
        cen_wh = cen_wh_from_tl_br(*np.hstack((tl, br)))
        r_det[3:11] = np.hstack((tl, br, cen_wh))
    return np.round(r_dets).astype(np.int64)


def rectify_tracks(tracks, cameraMatrix, distCoeffs, R, r_P):
    frame_numbers = np.unique(tracks[:, 1])  # sort
    step = min(np.diff(frame_numbers))
    start = min(frame_numbers)
    stop = max(frame_numbers)

    r_tracks = []
    for frame_number in tqdm(range(start, stop + 1, step)):
        if frame_number % step != 0:
            continue

        tracks[:, 2] = tracks[:, 0]  # dets or tracks used as dets
        dets = tracks[tracks[:, 1] == frame_number]

        r_dets = rectify_detections(dets, cameraMatrix, distCoeffs, R, r_P)
        r_tracks.append(r_dets)
    return np.concatenate(r_tracks, axis=0)


# Postprocess
# ===========
def postprocess_tracks(tracks):
    # remove static tracks, remove short tracks, reindexing, interpolate tracks
    tracks = tracks.copy()
    tracks = pp.remove_static_tracks(tracks, window_size=16, move_threshold=10)
    tracks = pp.remove_short_tracks(tracks, min_track_length=16)
    tracks = pp.reindex_tracks(tracks)
    tracks = pp.interpolate_tracks_when_missing_frames(tracks)
    return tracks


# Evaluation
# ==========
def match_gt_stereo_tracks(tracks1, tracks2):
    """match two ground truth tracks
    rectified coordinates should be given.
    return list[tuple(gid1, gid2)]
    >>> gtracks1 = rectify_tracks(gtracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
    >>> gtracks2 = rectify_tracks(gtracks2, cameraMatrix2, distCoeffs2, R2, r_P2)
    >>> g1g2_tids = match_gt_stereo_tracks(gtracks1, gtracks2)
    """
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
            # c1 = normalize_curve(track1[:, 7:9])
            # c2 = normalize_curve(track2[:, 7:9])
            # dist = curve_distance(c1, c2)
            n_points = track1.shape[0]
            dist = np.mean(abs(track1[:, 8] - track2[:, 8])) / n_points
            all_dists[(tid1, tid2)] = round(dist, 3)

    # n-n matching
    i2tids1 = {k: v for k, v in enumerate(tids1)}
    i2tids2 = {k: v for k, v in enumerate(tids2)}
    dist_mat = np.zeros((len(tids1), len(tids2)))
    for i, tid1 in enumerate(tids1):
        for j, tid2 in enumerate(tids2):
            dist_mat[i, j] = all_dists.get((tid1, tid2), max_dist)
    rows, cols = linear_sum_assignment(dist_mat)
    matched_tids = [(i2tids1[r], i2tids2[c]) for r, c in zip(rows, cols)]
    """
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
    """
    return matched_tids


# match with ground truth (for evaluation)
def match_tracks_with_gt(tracks, gtracks, step=8):
    """
    gid2tids: dict[list]. ground truth id to matched tids
    frames_tids: list[list]: each element: start frame, end frame, gt id, track id
    """
    gid2tids = defaultdict(list)
    frames_tids = []
    for gtid in np.unique(gtracks[:, 0]):
        gtrack = gtracks[(gtracks[:, 0] == gtid)]
        tid2frame = defaultdict(list)  # gtid to frame number
        for det in gtrack[::step]:
            tid = pp.tid_from_xyf(tracks, det[7], det[8], det[1], thrs=5)
            if tid is not None:
                tid2frame[tid].append(det[1])
        # take min and max
        for tid, frames in tid2frame.items():
            if min(frames) != max(frames):
                gid2tids[gtid].append(tid)
                frames_tids.append([min(frames), max(frames), gtid, tid])
    return gid2tids, frames_tids


# Debugging
# =========
def plot_matched_tracks_with_gt(tracks, gtracks, gtid, tids):
    gtrack = gtracks[(gtracks[:, 0] == gtid)]
    plt.figure()
    for tid in tids:
        track = tracks[(tracks[:, 0] == tid)]
        plt.plot(track[:, 7], track[:, 8], "o--", label=str(tid))
    plt.plot(gtrack[:, 7], gtrack[:, 8], "*-", color="gray", alpha=0.3)
    plt.legend()
    plt.title(f"{gtid}->{tids}")


# Merging
# =======
def merge_by_mached_tids(input_list):
    """
    If the matched track ids are the same merge them even if there is a gap.
    N.B. Each item of the list is [start_frame, end_frame, tid1, tid2]
    e.g. if [0, 400, 1, 4] and [400, 1000, 1, 4] then [0, 1000, 1, 4]
    e.g. if [0, 400, 1, 4] and [600, 1000, 1, 4] then [0, 1000, 1, 4]
    """
    # Create a dictionary to group lists by their last two elements
    merge_dict = {}
    for item in input_list:
        key = tuple(item[2:4])  # The key is the tuple of the last two elements
        if key not in merge_dict:
            merge_dict[key] = item
        else:
            # Merge the lists by updating the second element
            merge_dict[key][1] = item[1]

    # Extract the merged lists from the dictionary
    merged_list = list(merge_dict.values())

    return merged_list


# Visualization
# =============
def put_bbox_in_image(image, x_tl, y_tl, x_br, y_br, color, text, black=True):
    cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), color=color, thickness=1)
    if black:
        color = (0, 0, 0)
    # font name, font scale, color, thinckness, line type
    cv2.putText(image, text, (x_tl, y_tl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)


def draw_matches(image1, image2, dets1, dets2, color):
    combined_image = np.hstack((image1, image2))
    for d1, d2 in zip(dets1, dets2):
        start_point = (int(d1[7]), int(d1[8]))
        end_point = (int(d2[7]) + image1.shape[1], int(d2[8]))
        cv2.line(combined_image, start_point, end_point, (0, 255, 0), 1)

    for det in dets1:
        x_tl, y_tl, x_br, y_br = det[3:7]
        put_bbox_in_image(
            combined_image,
            int(x_tl),
            int(y_tl),
            int(x_br),
            int(y_br),
            color,
            f"{det[0]}",
        )

    for det in dets2:
        x_tl, y_tl, x_br, y_br = det[3:7]
        put_bbox_in_image(
            combined_image,
            int(x_tl) + image1.shape[1],
            int(y_tl),
            int(x_br) + image1.shape[1],
            int(y_br),
            color,
            f"{det[0]}",
        )

    return combined_image


def create_stereo_image_with_matches(
    image1, image2, dets1, dets2, frame, frame_matches
):
    """
    frame_matches: list[list], each item [start_frame, end_frame, tid1, tid2]
    """
    # find matches.
    tids1 = dets1[:, 0]
    tids2 = dets2[:, 0]
    matches = {
        frame_match[2]: frame_match[3]
        for frame_match in frame_matches
        if frame_match[0] <= frame < frame_match[1]
        and frame_match[2] in tids1
        and frame_match[3] in tids2
    }

    if not matches:
        return None

    # ordered based on matches
    mdets1 = np.array([dets1[dets1[:, 0] == tid1][0] for tid1 in matches.keys()])
    mdets2 = np.array([dets2[dets2[:, 0] == tid2][0] for tid2 in matches.values()])

    # plot and save results
    color = (0, 0, 255)
    combined_image = draw_matches(image1, image2, mdets1, mdets2, color)
    return combined_image


def save_stereo_images_with_matches_as_video(
    save_video_file,
    vid_path1,
    vid_path2,
    tracks1,
    tracks2,
    frame_matches,
    st_frame=0,
    en_frame=None,
    step=1,
    fps=30,
):
    save_video_file = Path(save_video_file)
    save_video_file.parent.mkdir(parents=True, exist_ok=True)

    vc1 = cv2.VideoCapture(str(vid_path1))
    vc2 = cv2.VideoCapture(str(vid_path2))
    if not en_frame:
        en_frame = (
            int(
                min(
                    vc1.get(cv2.CAP_PROP_FRAME_COUNT), vc2.get(cv2.CAP_PROP_FRAME_COUNT)
                )
            )
            - 1
        )  # total number of frames -1

    width = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH) + vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
    assert vc1.get(cv2.CAP_PROP_FRAME_HEIGHT) == vc2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"XVID") with .avi
    out = cv2.VideoWriter(str(save_video_file), fourcc, fps, (width, height))

    vc2.set(cv2.CAP_PROP_POS_FRAMES, st_frame)
    vc1.set(cv2.CAP_PROP_POS_FRAMES, st_frame)
    for frame in tqdm(range(st_frame, en_frame)):
        ret1, image1 = vc1.read()
        ret2, image2 = vc2.read()
        if not ret1 or not ret2:
            break
        if frame % step != 0:
            continue
        dets1 = tracks1[tracks1[:, 1] == frame]
        dets2 = tracks2[tracks2[:, 1] == frame]
        if (dets1.size == 0) or (dets2.size == 0):
            continue
        combined_image = create_stereo_image_with_matches(
            image1, image2, dets1, dets2, frame, frame_matches
        )
        out.write(combined_image)

    out.release()


def save_stereo_images_with_matches_as_images(
    save_path,
    vid_path1,
    vid_path2,
    tracks1,
    tracks2,
    frame_matches,
    st_frame=0,
    en_frame=None,
    step=1,
):
    save_path.mkdir(parents=True, exist_ok=True)

    vc1 = cv2.VideoCapture(str(vid_path1))
    vc2 = cv2.VideoCapture(str(vid_path2))
    if not en_frame:
        en_frame = (
            int(
                min(
                    vc1.get(cv2.CAP_PROP_FRAME_COUNT), vc2.get(cv2.CAP_PROP_FRAME_COUNT)
                )
            )
            - 1
        )  # total number of frames -1
    vc2.set(cv2.CAP_PROP_POS_FRAMES, st_frame)
    vc1.set(cv2.CAP_PROP_POS_FRAMES, st_frame)
    for frame in tqdm(range(st_frame, en_frame)):
        _, image1 = vc1.read()
        _, image2 = vc2.read()
        if frame % step != 0:
            continue
        dets1 = tracks1[tracks1[:, 1] == frame]
        dets2 = tracks2[tracks2[:, 1] == frame]
        if (dets1.size == 0) or (dets2.size == 0):
            continue
        combined_image = create_stereo_image_with_matches(
            image1, image2, dets1, dets2, frame, frame_matches
        )
        cv2.imwrite(f"{save_path}/frame_{frame:06d}.jpg", combined_image)


def save_images_as_video(vid_file, image_path, fps, width, height):
    """
    The same code as ffmpeg but get 3 time more storage.
    # ffmpeg -framerate 30 -pattern_type glob -i "frame_*.jpg" -c:v libx264 -pix_fmt yuv420p output.mp4
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"XVID") with .avi
    out = cv2.VideoWriter(str(vid_file), fourcc, fps, (width, height))
    image_files = sorted(
        image_path.glob("frame_*jpg"), key=lambda x: int(x.stem.split("_")[1])
    )
    for image_file in tqdm(image_files):
        image = cv2.imread(str(image_file))
        out.write(image)
    out.release()


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
mat_file = Path(inputs.mat_file)
vid_path1 = Path(inputs.vid_path1)
vid_path2 = Path(inputs.vid_path2)
track_file1 = Path(inputs.track_file1)
track_file2 = Path(inputs.track_file2)
if not inputs.gt_track_file1:
    gt_track_file1 = track_file1
    gt_track_file2 = track_file2
else:
    gt_track_file1 = Path(inputs.gt_track_file1)
    gt_track_file2 = Path(inputs.gt_track_file2)


# parameters
# ==========
max_dist = 100
# fmt: off
gt_matches = {3:0, 5:1, 4:2, 2:3, 8:4, 0:5, 1:6, 6:7, 7:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
# fmt: on

gtracks1 = da.load_tracks_from_mot_format(gt_track_file1)
gtracks2 = da.load_tracks_from_mot_format(gt_track_file2)
tracks1 = da.load_tracks_from_mot_format(track_file1)
tracks2 = da.load_tracks_from_mot_format(track_file2)

dd = loadmat(mat_file)
vc1 = cv2.VideoCapture(str(vid_path1))


distCoeffs1 = deepcopy(dd["distortionCoefficients1"])[0].astype(np.float64)
distCoeffs2 = deepcopy(dd["distortionCoefficients2"])[0].astype(np.float64)
cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"]).astype(np.float64)
cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"]).astype(np.float64)
R = deepcopy(dd["rotationOfCamera2"]).astype(np.float64)
T = deepcopy(dd["translationOfCamera2"]).T.astype(np.float64)
cameraMatrix1[0:2, 2] += 1
cameraMatrix2[0:2, 2] += 1
# Projection matrices
P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
P2 = np.dot(cameraMatrix2, np.hstack((R, T))).astype(np.float64)

_, image1 = vc1.read()
vc1.release()
image_size = image1.shape[1], image1.shape[0]
R1, R2, r_P1, r_P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
    imageSize=image_size,
    R=R,
    T=T,
)  # , flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)


# postprocess tracks
# =====
tracks1 = postprocess_tracks(tracks1)
tracks2 = postprocess_tracks(tracks2)
# from tracking import visualize
# visualize.save_images_with_tracks(
#                 Path("/home/fatemeh/Downloads/fish/mot_data/images/129_1/pp"),
#                 vid_path1,
#                 tracks1,
#                 0,
#                 None,
#                 1,
#             )


# rectify
# =======
otracks1 = tracks1.copy()
otracks2 = tracks2.copy()
tracks1 = rectify_tracks(tracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
tracks2 = rectify_tracks(tracks2, cameraMatrix2, distCoeffs2, R2, r_P2)

# debuging
# ============
# for tid1, tid2 in gt_matches.items():
#     track1, track2 = pp.get_matching_frames_between_tracks(tracks1, tracks2, tid1, tid2)
#     plt.figure();plt.plot(track1[:,1], track1[:,8]-track2[:,8], '*-')
#     plt.title(f'{tid1}:{tid2}')

# evaluation
# ==========
# # g_frame_matches(g1->g2), frames_gtids1(g1->t1), frames_gtids2(g2->t2) -> (t1->t2)

# # g1->t1, g2->t2
# gid2tids1, frames_gtids1 = match_tracks_with_gt(tracks1, gtracks1)
# gid2tids2, frames_gtids2 = match_tracks_with_gt(tracks2, gtracks2)
# # [plot_matched_tracks_with_gt(tracks2, gtracks2, k, v) for k, v in gid2tids1.items()]

# # g1->g2
# rgtracks1 = rectify_tracks(gtracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
# rgtracks2 = rectify_tracks(gtracks2, cameraMatrix2, distCoeffs2, R2, r_P2)
# g1g2_tids = match_gt_stereo_tracks(rgtracks1, rgtracks2)
# g_frame_matches = []
# for tid1, tid2 in g1g2_tids: #gt_matches.items():
#     track1, track2 = pp.get_matching_frames_between_tracks(
#         gtracks1, gtracks2, tid1, tid2
#     )
#     g_frame_matches.append([min(track1[:, 1]), max(track1[:, 1]), tid1, tid2])
# print(sorted(gt_matches.items(), key=lambda x:x[0]))
# print(g1g2_tids)
# print(g_frame_matches)

# tracklet matching
# =================
start, end, step = 0, 3117, 240  # 200

# for debuging
# tid1 = 3
# track1 = tracks1[tracks1[:, 0] == tid1]
# tracklet1 = cut_track_into_tracklets(track1, start, end, step)
# for st in range(start, end + 1, step):
#     tr = tracklet1[tracklet1[:, 2] == st]
#     plt.plot(tr[::16, 7], tr[::16, 8], "-*")


tracklets1 = cut_tracks_into_tracklets(tracks1, start, end, step)
tracklets2 = cut_tracks_into_tracklets(tracks2, start, end, step)

frame_matched_tids = dict()
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
            # c1 = normalize_curve(track1[:, 7:9])
            # c2 = normalize_curve(track2[:, 7:9])
            # dist = curve_distance(c1, c2)
            n_points = track1.shape[0]
            dist = np.mean(abs(track1[:, 8] - track2[:, 8])) / n_points
            all_dists[(tid1, tid2)] = round(dist, 3)

    # n-n matching
    i2tids1 = {k: v for k, v in enumerate(tids1)}
    i2tids2 = {k: v for k, v in enumerate(tids2)}
    dist_mat = np.zeros((len(tids1), len(tids2)))
    for i, tid1 in enumerate(tids1):
        for j, tid2 in enumerate(tids2):
            dist_mat[i, j] = all_dists.get((tid1, tid2), max_dist)
    rows, cols = linear_sum_assignment(dist_mat)
    matched_tids = [(i2tids1[r], i2tids2[c]) for r, c in zip(rows, cols)]
    frame_matched_tids[st] = matched_tids

    print("----> ", st)
    print(matched_tids)
    # for tid1, tid2 in frame_matched_tids[st]:
    #     track1, track2 = pp.get_matching_frames_between_tracks(tracks1, tracks2, tid1, tid2)
    #     plt.figure();plt.plot(track1[:, 7], track1[:, 8], "o--", label=str(tid1));plt.plot(track2[:, 7], track2[:, 8], "o--", label=str(tid2));plt.legend()


frame_matches = [
    [st, st + step, *match]
    for st, matches in frame_matched_tids.items()
    for match in matches
]

frame_matches1 = merge_by_mached_tids(frame_matches)

from scipy.ndimage import uniform_filter1d


# Apply a moving average filter to smooth the data
def get_matched_stereo_tracks(tracks1, tracks2, sframe, eframe, tid1, tid2):
    # TODO: rename both get_matched_stereo_tracks, get_matching_frames_between_tracks
    track1 = tracks1[
        (tracks1[:, 0] == tid1) & (tracks1[:, 1] >= sframe) & (tracks1[:, 1] < eframe)
    ]
    track2 = tracks2[
        (tracks2[:, 0] == tid2) & (tracks2[:, 1] >= sframe) & (tracks2[:, 1] < eframe)
    ]
    track1, track2 = pp.get_matching_frames_between_tracks(track1, track2, tid1, tid2)
    return track1, track2


def find_match_key(
    query, keys, tracks1, tracks2, thrs=20, frame_thrs=480, short_thrs=60
):
    """
    Finds a matching key for the given query based on various thresholds and conditions.

    Parameters:
        query (list): The query to match. e.g. [720, 2640, 0, 9] (start_frame, end_frame,
                      track_id1, track_id2)
        keys (list of lists): List of potential match keys. e.g. [[720, 2640, 0, 9],
                          [0, 480, 1, 3]]
        tracks1 (list): List of tracks from the first source.
        tracks2 (list): List of tracks from the second source.
        thrs (int, optional): Threshold for disparity difference. Default is 20.
        frame_thrs (int, optional): Threshold for frame difference. Default is 480.
        short_thrs (int, optional): Threshold for minimum track length. Default is 60.

    Returns:
        list or None: The matching key if found, otherwise None.
    """
    q_track1, q_track2 = get_matched_stereo_tracks(tracks1, tracks2, *query)
    if len(q_track1) == 0:
        return None

    q_disparities = q_track1[:, 7] - q_track2[:, 7]
    disp_diff = []

    for i, item in enumerate(keys):
        track1, track2 = get_matched_stereo_tracks(tracks1, tracks2, *item)
        if len(track1) == 0:
            continue
        if track1.shape[0] < short_thrs:
            continue
        if track1[0, 1] < q_track1[-1, 1]:
            continue
        if q_track1[-1, 1] >= track1[-1, 1]:
            continue
        if (track1[0, 1] - q_track1[-1, 1]) > frame_thrs:
            continue

        disparities = track1[:, 7] - track2[:, 7]
        abs_disp = abs(disparities[0] - q_disparities[-1])
        if abs_disp > thrs:
            continue

        if (query[2] == item[2]) or (query[3] == item[3]):
            return item

        disp_diff.append([i, abs_disp])

    if not disp_diff:
        return None
    if len(disp_diff) == 1:
        ind = disp_diff[0][0]
        return keys[ind]

    return None  # Ambiguous case with more than one possible match


"""
inds = [i for i, d in disp_diff if d<10]
cand_keys = [query] + [keys[k] for k in inds]
plt.figure()
for item in [[0, 2400, 2, 8], [2640, 3120, 2, 9], [0, 2400, 5, 7], [2400, 3120, 18, 7]]:#cand_keys:
    track1, track2 = get_matched_stereo_tracks(tracks1, tracks2, *item)
    plt.plot(track1[:,1], track1[:,7]-track2[:,7], '-*', label=f"{item[2]}:{item[3]}")
plt.legend()
"""

tracks1 = otracks1.copy()
tracks2 = otracks2.copy()
tracks1 = rectify_tracks(tracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
tracks2 = rectify_tracks(tracks2, cameraMatrix2, distCoeffs2, R2, r_P2)

all_checked = []
remained = frame_matches1.copy()
frame_matches2 = []
for i in range(start, end + 1, step):
    # Collect queries for the current frame index
    queries = [item for item in frame_matches1 if item[0] == i]

    # Include the last matched items from frame_matches2
    last_matches = [item[-1] for item in frame_matches2 if len(item) > 1]
    queries += last_matches

    # Remove already matched items from queries
    queries = [item for item in queries if item not in all_checked]

    # Remove queries from the remained list
    remained = [item for item in remained if item not in queries]

    # Add queries to all_checked
    all_checked.extend(queries)

    if not queries:
        continue

    for query in queries:
        matched_key = find_match_key(query, remained, tracks1, tracks2)

        if not matched_key:
            # If no match found, add query to frame_matches2 if not already present
            if not any(query in group for group in frame_matches2):
                frame_matches2.append([query])
            continue

        # Remove matched key from remained
        remained.remove(matched_key)

        # Add matched key to the corresponding group in frame_matches2
        query_found = False
        for group in frame_matches2:
            if query in group:
                group.append(matched_key)
                query_found = True
                break

        if not query_found:
            frame_matches2.append([query, matched_key])

"""
# Tobe removed. Just for debugging.
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
assert len(expected) == len(frame_matches2)
for exp, act in zip(expected, frame_matches2):
    assert exp == act
"""
print(frame_matches2)

# plt.figure()
# prev_disp = None
# for item in [[0, 1200, 8, 5], [1200, 2400, 8, 11], [2400, 3120, 8, 15]]:
#     track1, track2 = get_matched_stereo_tracks(tracks1, tracks2, *item)
#     disparity = track1[:,7]-track2[:,7]
#     smoothed_disp = uniform_filter1d(disparity, size=4*16)
#     if prev_disp is not None:
#         print(smoothed_disp[-1] - prev_disp)
#     prev_disp = smoothed_disp[-1]
#     plt.plot(track1[:,1], disparity, '*')
#     plt.plot(track1[:,1], smoothed_disp, 'r-')


# save_stereo_images_with_matches_as_images(
#     save_video_file/"tmp", vid_path1, vid_path2, otracks1, otracks2, frame_matches1, 0, None, 8
# )
# save_images_as_video(save_video_file, save_video_file/"tmp", 30, 3840, 1080)
# save_stereo_images_with_matches_as_video(
#     save_video_file,
#     vid_path1,
#     vid_path2,
#     otracks1,
#     otracks2,
#     frame_matches1,
#     inputs.start_frame,
#     inputs.end_frame,
#     inputs.step,
#     inputs.fps,
# )
print("======")
