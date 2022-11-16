from data_association import *

# from stereo import *
from visualize import *


result_folder = Path("/home/fatemeh/results/dataset1")
data_folder = Path("/home/fatemeh/data/dataset1")
det_folder1 = data_folder / "cam1_rect_labels"
det_folder2 = data_folder / "cam2_rect_labels"
filename_fixpart1 = "12_07_22_1_C_GH040468_1_cam1_rect"
filename_fixpart2 = "12_07_22_1_D_GH040468_1_cam2_rect"

# result_folder = Path("/home/fatemeh/results/dataset2")
# data_folder = Path("/home/fatemeh/data/dataset2")
# det_folder1 = data_folder / "cam1_labels"
# det_folder2 = data_folder / "cam2_labels"
# filename_fixpart1 = "10_07_22_A_GH040718_1_cam1"
# filename_fixpart2 = "10_07_22_B_GH040718_1_cam2"

vc1 = cv2.VideoCapture((data_folder / f"{filename_fixpart1}.mp4").as_posix())
vc2 = cv2.VideoCapture((data_folder / f"{filename_fixpart2}.mp4").as_posix())

height, width, total_no_frames, fps = get_video_parameters(vc1)

cam_id1 = 1
cam_id2 = 2
frame_number1 = 1
frame_number2 = 2
det_path1_c1 = det_folder1 / f"{filename_fixpart1}_{frame_number1}.txt"
det_path1_c2 = det_folder2 / f"{filename_fixpart2}_{frame_number1}.txt"
det_path2_c1 = det_folder1 / f"{filename_fixpart1}_{frame_number2}.txt"
det_path2_c2 = det_folder2 / f"{filename_fixpart2}_{frame_number2}.txt"

dets1_c1_disp = get_detections_with_disp(
    det_path1_c1, det_path1_c2, frame_number1, width, height, cam_id1
)
dets2_c1_disp = get_detections_with_disp(
    det_path2_c1, det_path2_c2, frame_number2, width, height, cam_id1
)
dets1_c2_disp = get_detections_with_disp(
    det_path1_c2, det_path1_c1, frame_number1, width, height, cam_id2
)
dets2_c2_disp = get_detections_with_disp(
    det_path2_c2, det_path2_c1, frame_number2, width, height, cam_id2
)

tracks1_disp = compute_tracks_with_disps(
    det_folder1,
    filename_fixpart1,
    det_folder2,
    filename_fixpart2,
    cam_id1,
    width,
    height,
    total_no_frames,
)
tracks2_disp = compute_tracks_with_disps(
    det_folder2,
    filename_fixpart2,
    det_folder1,
    filename_fixpart1,
    cam_id2,
    width,
    height,
    total_no_frames,
)

dets1_c1 = get_detections(det_path1_c1, frame_number1, width, height, cam_id1)
dets2_c1 = get_detections(det_path2_c2, frame_number1, width, height, cam_id1)

tracks1 = compute_tracks(
    det_folder1, filename_fixpart1, cam_id1, width, height, total_no_frames
)
tracks2 = compute_tracks(
    det_folder2, filename_fixpart2, cam_id2, width, height, total_no_frames
)

"""
dets3_c1_disp = get_detections_with_disp(det_folder1 / f"{filename_fixpart1}_{3}.txt", det_folder2 / f"{filename_fixpart2}_{3}.txt", 3, width, height, cam_id1)
tracks1_disp = compute_tracks_with_disps(det_folder1,filename_fixpart1,det_folder2,filename_fixpart2,cam_id1,width,height,2)
pred_dets = [
            track.predicted_loc
            for _, track in tracks1_disp.items()
            if track.status != Status.Stoped
        ]

ids1, ids2 = match_two_detection_sets(pred_dets, dets3_c1_disp)

frame2_c1, frame2_c2 = get_stereo_frames(2, vc1, vc2)
frame3_c1, frame3_c2 = get_stereo_frames(3, vc1, vc2)

_, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axs = _show_two_frames(axes, frame2_c1, frame3_c1)
_draw_detections_and_flows(dets2_c1_disp, axs[0])
_draw_detections_and_flows(dets3_c1_disp, axs[1])
for det in pred_dets:
    axs[1].plot([det.x], [det.y], "*r")
    axs[1].text(det.x, det.y, str(det.track_id), color="r", fontsize=12)
for id1, id2 in zip(ids1, ids2):
    axs[1].plot([pred_dets[id1].x, dets3_c1_disp[id2].x], [pred_dets[id1].y, dets3_c1_disp[id2].y], "-r")

"""
# (array([ 0,  1,  2,  3,  4,  5,  6,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#         36, 37, 38]),
#  array([ 3, 10,  4,  2,  0,  1, 11,  6,  5,  9, 16,  8, 12, 15,  7, 13, 17,
#         24, 22, 14, 20, 19, 32, 21, 26, 25, 27, 28, 30, 23, 29, 31, 33, 18,
#         34, 35, 36]))


def get_stereo_frames(frame_number, vc1, vc2):
    vc1.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    vc2.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    _, frame_c1 = vc1.read()
    _, frame_c2 = vc2.read()
    return frame_c1, frame_c2


"""
matches = compute_match_candidates(dets1, dets2, inverse=False)
show_detections_in_stereo(frame1, frame2, dets1, dets2, width)
for match in matches:
    disps = ""
    for candidate in match.candidates:
        disps += f"{candidate.det.det_id}:{str(candidate.disp)},"
    print(f"{match.target.det.det_id}, {match.target.det.y}: {disps}")
"""

# plot_frameid_y_for_stereo(tracks1, [15], tracks2, [16])

# save_tracks(result_folder / "tracks1.txt", tracks1)
# save_tracks(result_folder / "tracks2.txt", tracks2)

# all_matches = compute_possible_matches(tracks1, tracks2)
# all_matches_inv = compute_possible_matches(tracks2, tracks1)
# save_all_matches(result_folder / "matches.txt", all_matches, inverse=False)
# save_all_matches(result_folder / "matches.txt", all_matches_inv, inverse=True)
