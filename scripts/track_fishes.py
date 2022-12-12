import argparse
import os
import sys
from pathlib import Path

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import cv2
import matplotlib.pylab as plt

from tracking.data_association import (
    Point,
    compute_tracks,
    get_video_parameters,
    save_tracks_to_mot_format,
)
from tracking.visualize import get_frame, visualize_tracks_in_video


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-r",
        "--result_folder",
        help="Path to save the result",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--det_folder",
        help="Path where the detections are",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-v", "--video_file", help="Video file with full path", required=True, type=Path
    )
    parser.add_argument(
        "--save_name",
        help="The name of the file used to save tracks and video",
        required=False,
        type=str,
        default="tracks_test",
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
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    result_folder = args.result_folder.absolute()
    det_folder = args.det_folder.absolute()
    video_file = args.video_file.absolute()
    filename_fixpart = video_file.stem

    result_folder.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(video_file.as_posix())

    height, width, total_no_frames, fps = get_video_parameters(vc)
    if args.fps is None:
        args.fps = fps
    if args.total_no_frames is None:
        args.total_no_frames = total_no_frames

    cam_id = 1

    tracks = compute_tracks(
        det_folder, filename_fixpart, cam_id, width, height, total_no_frames
    )

    if args.video_bbox is None:
        top_left = Point(x=0, y=0)
        video_width = width
        video_height = height
    else:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(
            int, args.video_bbox.split(",")
        )
        top_left = Point(top_left_x, top_left_y)
        video_width = bottom_right_x - top_left_x
        video_height = bottom_right_y - top_left_y

    visualize_tracks_in_video(
        tracks,
        vc,
        result_folder / f"{args.save_name}.mp4",
        top_left,
        video_width,
        video_height,
        args.total_no_frames,
        fps=args.fps,
        show_det_id=False,
        black=True,
    )
    save_tracks_to_mot_format(result_folder / f"{args.save_name}", tracks)


if __name__ == "__main__":
    main()
