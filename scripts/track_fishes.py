import argparse
import sys
from pathlib import Path

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import cv2

from tracking.data_association import (
    Point,
    _reindex_tracks,
    _remove_short_tracks,
    compute_tracks,
    save_tracks_to_mot_format,
)
from tracking.visualize import get_video_parameters, plot_tracks_in_video


def parse_args():
    parser = argparse.ArgumentParser(description="Track fishes")
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
    result_folder = args.result_folder.expanduser()
    det_folder = args.det_folder.expanduser()
    video_file = args.video_file.expanduser()
    filename_fixpart = video_file.stem

    result_folder.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(video_file.as_posix())

    height, width, total_no_frames, fps = get_video_parameters(vc)
    if args.fps is None:
        args.fps = fps
    if args.total_no_frames is None:
        args.total_no_frames = total_no_frames

    tracks = compute_tracks(
        det_folder, filename_fixpart, width, height, end_frame=args.total_no_frames - 1
    )
    tracks = _reindex_tracks(_remove_short_tracks(tracks))

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

    plot_tracks_in_video(
        result_folder / f"{args.save_name}.mp4",
        tracks,
        vc,
        top_left,
        video_width,
        video_height,
        start_frame=0,
        end_frame=args.total_no_frames - 1,
        fps=args.fps,
        show_det_id=False,
        black=True,
    )
    save_tracks_to_mot_format(result_folder / f"{args.save_name}", tracks)


if __name__ == "__main__":
    main()
