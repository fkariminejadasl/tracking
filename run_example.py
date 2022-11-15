from data_association import *
from stereo import *
from visualize import *

# from utils import *

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


# # tracking
tracks1 = compute_tracks(det_folder1, filename_fixpart1, width, height, total_no_frames)
tracks2 = compute_tracks(det_folder2, filename_fixpart2, width, height, total_no_frames)


frame_number = 1
vc1.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
_, frame1 = vc1.read()
vc2.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
_, frame2 = vc2.read()

dets1 = get_detections(det_folder1/f"{filename_fixpart1}_{frame_number}.txt", frame_number, width, height)
dets2 = get_detections(det_folder2/f"{filename_fixpart2}_{frame_number}.txt", frame_number, width, height)
matches = compute_match_candidates(dets1, dets2, inverse=False)

draw_detections_in_stereo(frame1, frame2, dets1, dets2, width)

for match in matches:
    disps = ""
    for candidate in match.candidates:
        disps += f"{candidate.det.det_id}:{str(candidate.disp)},"
    print(f"{match.target.det.det_id}, {match.target.det.y}: {disps}")


# plot_frameid_y_for_stereo(tracks1, [15], tracks2, [16])

# save_tracks(result_folder / "tracks1.txt", tracks1)
# save_tracks(result_folder / "tracks2.txt", tracks2)

# all_matches = compute_possible_matches(tracks1, tracks2)
# all_matches_inv = compute_possible_matches(tracks2, tracks1)
# save_all_matches(result_folder / "matches.txt", all_matches, inverse=False)
# save_all_matches(result_folder / "matches.txt", all_matches_inv, inverse=True)
