import argparse
import sys
from pathlib import Path

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from tracking.data_association import (
    Point,
    _reindex_tracks,
    _remove_short_tracks,
    compute_tracks,
    make_array_from_tracks,
    save_tracks_to_mot_format,
)
from tracking.multi_stage_tracking import multistage_track, ultralytics_track
from tracking.visualize import (
    get_video_parameters,
    plot_tracks_in_video,
    save_images_with_tracks,
)


def process_config(config_path):
    with open(config_path, "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a config file.")
    parser.add_argument("config_file", type=Path, help="Path to the config file")

    args = parser.parse_args()
    config_path = args.config_file
    inputs = process_config(config_path)
    for key, value in inputs.items():
        print(f"{key}: {value}")
    inputs = SimpleNamespace(**inputs)

    vid_name = inputs.vid_name
    folder = inputs.folder
    start_frame = inputs.start_frame
    end_frame = inputs.end_frame
    step = inputs.step
    format = inputs.format
    track_method = inputs.track_method
    image_folder = inputs.image_folder
    det_checkpoint = Path(inputs.det_checkpoint)
    main_path = Path(inputs.main_path)
    config_file = Path(inputs.config_file)

    main_path = main_path / f"{folder}"
    save_name = f"{track_method}_8"

    if track_method == "ms":
        trks = multistage_track(
            main_path,
            image_folder,
            vid_name,
            start_frame,
            end_frame,
            step,
        )
    if track_method == "botsort" or track_method == "bytetrack":
        trks = ultralytics_track(
            main_path,
            image_folder,
            vid_name,
            start_frame,
            end_frame,
            step,
            det_checkpoint,
            config_file="botsort.yaml",
        )
    np.savetxt(main_path / f"{save_name}.txt", trks, delimiter=",", fmt="%d")
    save_tracks_to_mot_format(main_path / save_name, trks[:, :11])
    save_images_with_tracks(
        main_path / save_name,
        main_path / f"vids/{vid_name}.mp4",
        trks,
        start_frame,
        end_frame,
        step,
        format,
    )
