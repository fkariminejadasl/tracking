import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from tracking.data_association import load_tracks_from_mot_format
from tracking.visualize import plot_tracks_array_in_video


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

    if track_path.is_file():
        track_files = [track_path]
    if track_path.is_dir():
        track_files = track_path.glob("*")

    for track_file in track_files:
        if video_path.is_file():
            video_file = video_path
        else:
            video_file = video_path / f"{track_file.stem}.mp4"
        save_file = save_path / f"{track_file.stem}.mp4"

        if not save_file.exists():
            print(save_file.name)
            tracks = load_tracks_from_mot_format(track_file)
            plot_tracks_array_in_video(
                save_file,
                tracks,
                video_file,
                black=False,
            )


if __name__ == "__main__":
    inputs = parse_args()
    main(inputs)
