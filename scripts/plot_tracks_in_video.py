import argparse
import sys
from pathlib import Path

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import cv2

from tracking.data_association import Point, load_tracks_from_mot_format
from tracking.visualize import get_video_parameters, plot_tracks_array_in_video


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tracks bounding boxes in video")
    parser.add_argument(
        "-t",
        "--track_file",
        help="Track file",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-v",
        "--video_file",
        help="Video file extension '.mp4'",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--save_file",
        help="The resulting video with or without full path.",
        required=True,
        type=Path,
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
    track_file = args.track_file.expanduser()
    video_file = args.video_file.expanduser()
    save_file = args.save_file.expanduser()

    save_file.parent.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(video_file.as_posix())

    height, width, total_no_frames, fps = get_video_parameters(vc)
    if args.fps is None:
        args.fps = fps
    if args.total_no_frames is None:
        args.total_no_frames = total_no_frames

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

    tracks = load_tracks_from_mot_format(track_file)
    plot_tracks_array_in_video(
        save_file,
        tracks,
        vc,
        top_left,
        video_width,
        video_height,
        end_frame=args.total_no_frames - 1,
        fps=args.fps,
        show_det_id=False,
        black=False,
    )


if __name__ == "__main__":
    main()
