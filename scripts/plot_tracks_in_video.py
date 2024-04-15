import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from tracking.data_association import load_tracks_from_mot_format
from tracking.visualize import plot_tracks_array_in_video, save_images_with_tracks


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
    track_path = Path(inputs.track_path)
    video_path = Path(inputs.video_path)
    save_path = Path(inputs.save_path)

    assert track_path.exists(), f"Error: {track_path} doesn't exist!"
    if track_path.is_file():
        track_files = [track_path]
    if track_path.is_dir():
        track_files = track_path.glob("*")

    for track_file in track_files:
        track_name = f"{track_file.stem}"
        if video_path.is_file():
            video_file = video_path
        else:
            video_file = video_path / f"{track_name}.mp4"

        print(track_name)
        tracks = load_tracks_from_mot_format(track_file)
        frame_numbers = np.unique(tracks[:, 1])
        start_frame = frame_numbers[0]
        end_frame = frame_numbers[-1]
        step = min(np.diff(frame_numbers))
        if inputs.save_images:
            save_file = save_path
        else:
            save_file = save_path / f"{track_name}.mp4"
        if save_file.exists():
            continue

        if inputs.save_images:
            save_images_with_tracks(
                save_path,
                video_file,
                tracks,
                start_frame,
                end_frame,
                step,
            )
        else:
            plot_tracks_array_in_video(
                save_file,
                tracks,
                video_file,
                start_frame,
                end_frame,
                step,
                black=False,
            )


if __name__ == "__main__":
    inputs = parse_args()
    main(inputs)
