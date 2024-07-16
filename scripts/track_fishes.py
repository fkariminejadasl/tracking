import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from tracking import utils
from tracking.data_association import hungarian_track, save_tracks_to_mot_format
from tracking.multi_stage_tracking import (
    multistage_track,
    ultralytics_detect_video,
    ultralytics_track_video,
)
from tracking.visualize import get_video_parameters


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


def main(inputs):
    start_frame = inputs.start_frame
    end_frame = inputs.end_frame
    step = inputs.step
    track_method = inputs.track_method
    det_checkpoint = Path(inputs.det_checkpoint)
    main_save_path = Path(inputs.save_path)
    track_config_file = inputs.track_config_file
    video_path = Path(inputs.video_path)

    track_config_file = (
        Path(inputs.track_config_file)
        if inputs.track_config_file
        else Path(f"{track_method}.yaml")
    )

    # N.B. in yolov8, if image list is empty, it will donwload own bus.jpg.
    if video_path.is_dir():
        video_files = list(video_path.glob("*mp4"))
    elif video_path.is_file():
        video_files = [video_path]
    else:
        sys.exit(f"Error: The path '{video_path}' is neither a file nor a directory.")

    for video_file in video_files:
        stime = time.time()

        vid_name = video_file.stem
        exp_name = inputs.save_name
        save_name = f"{vid_name}_{track_method}_{exp_name}"
        save_path = main_save_path / f"{track_method}_{exp_name}"
        txt_path = save_path / f"extras"
        txt_path.mkdir(parents=True, exist_ok=True)
        txt_file = txt_path / f"{save_name}.txt"
        mot_file = save_path / f"mots/{save_name}.txt"
        if txt_file.exists() and mot_file.exists():
            continue
        print(vid_name)

        height, width, total_no_frames, _ = get_video_parameters(video_file)
        if end_frame is None:
            end_frame = total_no_frames - 1

        if track_method == "ms":
            if inputs.dets_path:  #
                dets_path = Path(inputs.dets_path) / f"{vid_name}_dets.zip"
            if (not inputs.dets_path) or (not dets_path.is_file()):  # None
                dets_path = main_save_path / f"yolov8/{vid_name}_dets.zip"
                print("=====> Detection")
                dets = ultralytics_detect_video(
                    video_file,
                    start_frame,
                    end_frame,
                    step,
                    det_checkpoint,
                )
                save_tracks_to_mot_format(dets_path, dets)

            print(f"=====> {track_method} tracking")
            trks = multistage_track(
                video_file,
                dets_path,
                vid_name,
                start_frame,
                end_frame,
                step,
            )

        if track_method == "botsort" or track_method == "bytetrack":
            print(f"=====> {track_method} tracking")
            trks = ultralytics_track_video(
                video_file,
                start_frame,
                end_frame,
                step,
                det_checkpoint,
                config_file=track_config_file,
            )

        if track_method == "hungarian":
            print(f"=====> {track_method} tracking")
            # TODO broken: only works with detection files (txt) not zip file
            trks = hungarian_track(
                dets_path,
                vid_name,
                width,
                height,
                start_frame,
                end_frame,
                step,
            )

        np.savetxt(txt_file, trks, delimiter=",", fmt="%d")
        # save_tracks_to_mot_format(save_path / f"mots/{save_name}", trks[:, :11])
        utils.track_file_to_mot_file(txt_file, mot_file)

        with open(save_path / f"{track_method}_{exp_name}_meta.csv", "a") as rfile:
            elapse_time = time.time() - stime
            rfile.write(
                f"{vid_name},{track_method},{total_no_frames},{elapse_time:.1f}\n"
            )


if __name__ == "__main__":
    inputs = parse_args()
    main(inputs)
