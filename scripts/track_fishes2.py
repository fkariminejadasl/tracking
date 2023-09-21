import argparse
import sys
from pathlib import Path

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from types import SimpleNamespace

import numpy as np
import yaml

from tracking.data_association import hungarian_track, save_tracks_to_mot_format
from tracking.multi_stage_tracking import (
    multistage_track,
    ultralytics_detect,
    ultralytics_detect_video,
    ultralytics_track,
    ultralytics_track_video,
)
from tracking.visualize import (
    get_video_parameters,
    save_images_of_video,
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

    start_frame = inputs.start_frame
    end_frame = inputs.end_frame
    step = inputs.step
    format = inputs.format
    track_method = inputs.track_method
    image_folder = inputs.image_folder
    det_checkpoint = Path(inputs.det_checkpoint)
    main_path = Path(inputs.main_path)
    track_config_file = inputs.track_config_file
    video_file = Path(inputs.video_file)
    dets_path = inputs.dets_path
    save_images = inputs.save_images
    save_name = inputs.save_name

    height, width, total_no_frames, _ = get_video_parameters(video_file)
    if end_frame is None:
        end_frame = total_no_frames - 1
    is_valid = False
    if dets_path:
        dets_path = Path(dets_path)
        is_valid = dets_path.exists()
    save_name = f"{track_method}_{save_name}"
    vid_name = video_file.stem
    track_config_file = (
        Path(inputs.track_config_file)
        if inputs.track_config_file
        else Path(f"{track_method}.yaml")
    )

    # TODO this is super ugly. If image_folder is not empty and contain no image,
    # it will donwload own bus.jpg.
    image_path = main_path / image_folder
    image_path.mkdir(parents=True, exist_ok=True)
    is_empty = not any(image_path.iterdir())
    if is_empty and save_images:
        print("=====> Save Images")
        save_images_of_video(
            main_path / f"{image_folder}",
            video_file,
            start_frame,
            end_frame,
            step,
            format,
        )

    if track_method == "ms":
        # TODO is_valid for other path. Put them in the beginning
        if isinstance(dets_path, Path) and dets_path.is_dir():
            is_valid = any(dets_path.iterdir())
        if not dets_path or not is_valid:
            dets_path = main_path / f"{save_name}_dets"
            print("=====> Detection")
            if is_empty:
                dets = ultralytics_detect_video(
                    video_file,
                    start_frame,
                    end_frame,
                    step,
                    det_checkpoint,
                )
            else:
                dets = ultralytics_detect(
                    image_path,
                    vid_name,
                    start_frame,
                    end_frame,
                    step,
                    det_checkpoint,
                )
            save_tracks_to_mot_format(dets_path, dets[:, :11])
            # TODO solve it in save_tracks_to_mot_format to get .zip file
            dets_path = main_path / f"{save_name}_dets.zip"

        print(f"=====> {track_method} tracking")
        trks = multistage_track(
            image_path,
            dets_path,
            vid_name,
            start_frame,
            end_frame,
            step,
        )

    if track_method == "botsort" or track_method == "bytetrack":
        print(f"=====> {track_method} tracking")
        if is_empty:
            print("====> read from video")
            trks = ultralytics_track_video(
                video_file,
                start_frame,
                end_frame,
                step,
                det_checkpoint,
                config_file=track_config_file,
            )
        else:
            trks = ultralytics_track(
                image_path,
                vid_name,
                start_frame,
                end_frame,
                step,
                det_checkpoint,
                config_file=track_config_file,
            )

    if track_method == "hungarian":
        print(f"=====> {track_method} tracking")
        # TODO only works with detection files
        trks = hungarian_track(
            dets_path,
            vid_name,
            width,
            height,
            start_frame,
            end_frame,
            step,
        )

    save_tracks_to_mot_format(main_path / save_name, trks[:, :11])
    if save_images:
        print("=====> Save Tracks in Images")
        np.savetxt(main_path / f"{save_name}.txt", trks, delimiter=",", fmt="%d")
        save_images_with_tracks(
            main_path / save_name,
            video_file,
            trks,
            start_frame,
            end_frame,
            step,
            format,
        )

    # cleanup
    if is_empty and not save_images:
        image_path.rmdir()
