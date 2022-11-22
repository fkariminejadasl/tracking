from data_association import *
from data_association import (_track_current_unmatched, _track_matches,
                              _track_predicted_unmatched)
# from stereo import *
from visualize import *
from visualize import _draw_detections_and_flows, _show_two_frames

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

# result_folder = Path("/home/fatemeh/results/dataset3")
# data_folder = Path("/home/fatemeh/data/dataset3")
# det_folder1 = data_folder / "cam1_labels"
# det_folder2 = data_folder / "cam2_labels"
# filename_fixpart1 = "12_07_22_1_F_GH040291_1_cam1"
# filename_fixpart2 = "12_07_22_1_H_GH040291_1_cam2"

# result_folder = Path("/home/fatemeh/results/dataset4")
# data_folder = Path("/home/fatemeh/data/dataset4")
# det_folder1 = data_folder / "cam1_rect_labels"
# det_folder2 = data_folder / "cam2_rect_labels"
# filename_fixpart1 = "04_07_22_F_1_rect"
# filename_fixpart2 = "04_07_22_G_1_rect"

result_folder = Path("/home/fatemeh/results/dataset5")
data_folder = Path("/home/fatemeh/data/dataset5")
det_folder1 = data_folder / "cam1_labels"
det_folder2 = data_folder / "cam2_labels"
filename_fixpart1 = "04_07_22_F_2_rect_valid"
filename_fixpart2 = "04_07_22_G_2_rect_valid"

result_folder.mkdir(parents=True, exist_ok=True)
vc1 = cv2.VideoCapture((data_folder / f"{filename_fixpart1}.mp4").as_posix())
vc2 = cv2.VideoCapture((data_folder / f"{filename_fixpart2}.mp4").as_posix())

height, width, total_no_frames, fps = get_video_parameters(vc1)

cam_id1 = 1
cam_id2 = 2

tracks1 = compute_tracks(
    det_folder1, filename_fixpart1, cam_id1, width, height, total_no_frames
)
tracks2 = compute_tracks(
    det_folder2, filename_fixpart2, cam_id2, width, height, total_no_frames
)
"""
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
fn = 147
# dets = get_detections_with_disp(
#     det_folder1 / f"{filename_fixpart1}_{fn}.txt",
#     det_folder2 / f"{filename_fixpart2}_{fn}.txt",
#     fn,
#     width,
#     height,
#     cam_id1,
# )
# dets_prev = get_detections_with_disp(
#     det_folder1 / f"{filename_fixpart1}_{fn-1}.txt",
#     det_folder2 / f"{filename_fixpart2}_{fn-1}.txt",
#     fn - 1,
#     width,
#     height,
#     cam_id1,
# )
# tracks = compute_tracks_with_disps(
#     det_folder1,
#     filename_fixpart1,
#     det_folder2,
#     filename_fixpart2,
#     cam_id1,
#     width,
#     height,
#     fn - 1,
# )
dets = get_detections(
    det_folder1 / f"{filename_fixpart1}_{fn}.txt",
    width,
    height,
    cam_id1,
)
dets_prev = get_detections(
    det_folder1 / f"{filename_fixpart1}_{fn-1}.txt",
    width,
    height,
    cam_id1,
)
tracks = compute_tracks(
    det_folder1,
    filename_fixpart1,
    cam_id1,
    width,
    height,
    fn - 1,
)
pred_dets = [
    track.predicted_loc for _, track in tracks.items() if track.status != Status.Stoped
]
track_dets_prev = find_detectios_in_tracks_by_frame_number(tracks, fn - 1)

pred_ids, ids = match_two_detection_sets(pred_dets, dets)

frame_prev = get_frame(fn - 1, vc1)
frame_curr = get_frame(fn, vc1)

_, axs = plt.subplots(1, 2, sharex=True, sharey=True)
_show_two_frames(axs, frame_prev, frame_curr)
_draw_detections_and_flows(dets_prev, axs[0])
_draw_detections_and_flows(dets, axs[1])
for track_id, det in track_dets_prev.items():
    axs[0].plot([det.x], [det.y], "*r")
    axs[0].text(det.x, det.y, str(track_id), color="r", fontsize=12)
for pred_det in pred_dets:
    axs[1].plot([pred_det.x], [pred_det.y], "*r")
    axs[1].text(pred_det.x, pred_det.y, str(pred_det.track_id), color="r", fontsize=12)
for id1, id2 in zip(pred_ids, ids):
    axs[1].plot([pred_dets[id1].x, dets[id2].x], [pred_dets[id1].y, dets[id2].y], "-r")
plt.show(block=False)

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
