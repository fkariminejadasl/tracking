import argparse
from pathlib import Path
import sys
import os

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import cv2
import matplotlib.pylab as plt
from tracking.data_association import (
    _track_current_unmatched,
    _track_matches,
    _track_predicted_unmatched,
    get_video_parameters,
    compute_tracks,
    get_detections,
    Status,
    find_detectios_in_tracks_by_frame_number,
    match_two_detection_sets,
    Point,
    save_tracks,
)

from tracking.visualize import (
    _draw_detections_and_flows,
    _show_two_frames,
    get_frame,
    visualize_tracks_in_video,
)


def parse_args():
    # result_folder = Path("/home/fatemeh/results/dataset5")
    # data_folder = Path("/home/fatemeh/data/dataset5")
    # filename_fixpart1 = "04_07_22_F_2_rect_valid"
    # filename_fixpart2 = "04_07_22_G_2_rect_valid"

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-r",
        "--result_folder",
        help="Path to save the result",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d", "--data_folder", help="Path where the data is", required=True, type=Path
    )
    parser.add_argument(
        "-f1",
        "--filename_fixpart1",
        help="Fix part of the name in detection file in cam1",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f2",
        "--filename_fixpart2",
        help="Fix part of the name in detection file in cam2",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--save_name",
        help="The name of the file used to save tracks and video",
        required=False,
        type=str,
        default="tracks1_test",
    )
    parser.add_argument(
        "--fps",
        help="frame per second",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--total_no_frames",
        help="total number of frames",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--video_bbox",
        help="""
        Crop and save the videos. Values are given by comma separated, 
        top_left.x,top_left.y,bottom_right.x,bottom_right.y.
        """,
        required=False,
        type=str,
        # default="270,100,1800,1200",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    det_folder1 = args.data_folder / "cam1_labels"
    det_folder2 = args.data_folder / "cam2_labels"

    args.result_folder.mkdir(parents=True, exist_ok=True)
    vc1 = cv2.VideoCapture(
        (args.data_folder / f"{args.filename_fixpart1}.mp4").as_posix()
    )
    vc2 = cv2.VideoCapture(
        (args.data_folder / f"{args.filename_fixpart2}.mp4").as_posix()
    )

    height, width, total_no_frames, fps = get_video_parameters(vc1)
    if args.fps is None:
        args.fps = fps
    if args.total_no_frames is None:
        args.total_no_frames = total_no_frames

    cam_id1 = 1
    cam_id2 = 2

    tracks1 = compute_tracks(
        det_folder1, args.filename_fixpart1, cam_id1, width, height, total_no_frames
    )
    tracks2 = compute_tracks(
        det_folder2, args.filename_fixpart2, cam_id2, width, height, total_no_frames
    )

    if args.video_bbox is None:
        top_left = Point(x=0, y=0)
        video_width = width
        video_height = height
    else:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, args.video_bbox.split(','))
        top_left = Point(top_left_x, top_left_y)
        video_width = bottom_right_x-top_left_x
        video_height = bottom_right_y-top_left_y
        

    visualize_tracks_in_video(
        tracks1,
        vc1,
        args.result_folder / f"{args.save_name}.mp4",
        top_left,
        video_width,
        video_height,
        args.total_no_frames,
        fps=args.fps,
        show_det_id=False,
        black=True,
    )
    save_tracks(args.result_folder / f"{args.save_name}.txt", tracks1)



if __name__ == "__main__":
    main()
