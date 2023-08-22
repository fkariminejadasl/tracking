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
    make_array_from_tracks,
    save_tracks_to_mot_format,
)
from tracking.visualize import (
    get_video_parameters,
    plot_tracks_in_video,
    save_images_with_tracks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Track fishes")
    parser.add_argument(
        "-r",
        "--save_path",
        help="Path to save the result",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--dets_path",
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
        "-sf",
        "--start_frame",
        help="starting frame. Frames counts from zero.",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "-ef",
        "--end_frame",
        help="The end frame. The counting of frames is zero-based.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--step",
        help="only track with this step size",
        required=False,
        type=int,
        default=8,
    )
    parser.add_argument(
        "--format",
        help="detection files are saved with this format. \
        e.g. name_0000123.txt the format is '06d'. \
        e.g. name_123.txt the format is ''",
        required=False,
        type=str,
        default="",
    )
    parser.add_argument(
        "--fps",
        help="frame per second",
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
    save_path = args.save_path.expanduser().resolve()
    dets_path = args.dets_path.expanduser().resolve()
    video_file = args.video_file.expanduser().resolve()
    filename_fixpart = video_file.stem
    save_path.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(video_file.as_posix())

    height, width, total_no_frames, fps = get_video_parameters(vc)
    if args.fps is None:
        args.fps = fps
    if args.end_frame is None:
        args.end_frame = total_no_frames - 1

    print(
        save_path,
        dets_path,
        video_file,
        f"end_frame={args.end_frame}",
        f"fps={args.fps}",
    )

    tracks = compute_tracks(
        dets_path,
        filename_fixpart,
        width,
        height,
        args.start_frame,
        args.end_frame,
        args.step,
        args.format,
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

    save_images_with_tracks(
        save_path / f"{args.save_name}",
        save_path / f"{video_file}",
        make_array_from_tracks(tracks),
        args.start_frame,
        args.end_frame,
        args.step,
        args.format,
    )

    plot_tracks_in_video(
        save_path / f"{args.save_name}.mp4",
        tracks,
        vc,
        top_left,
        video_width,
        video_height,
        args.start_frame,
        args.end_frame,
        args.step,
        fps=args.fps,
        show_det_id=False,
        black=True,
    )
    save_tracks_to_mot_format(save_path / f"{args.save_name}", tracks)


if __name__ == "__main__":
    main()
