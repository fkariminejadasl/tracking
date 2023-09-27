import argparse
import sys
from pathlib import Path

import numpy as np

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)


from tracking.data_association import load_tracks_from_mot_format
from tracking.stereo_gt import get_matched_track_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Match ground-truth stereo tracks")
    parser.add_argument(
        "-r",
        "--result_folder",
        help="Path to save the result",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--data_folder",
        help="Path where the tracks are",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-t1",
        "--tracks1",
        help="tracks1 in mot format",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-t2",
        "--tracks2",
        help="tracks1 in mot format",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save_name",
        help="The name of the file used to save matches and 3D coordinates",
        required=False,
        type=str,
        default="tracks_test",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    result_folder = args.result_folder.expanduser().resolve()
    data_folder = args.data_folder.expanduser().resolve()

    tracks1 = load_tracks_from_mot_format(Path(data_folder / args.tracks1))
    tracks2 = load_tracks_from_mot_format(Path(data_folder / args.tracks2))
    matches = np.array(get_matched_track_ids(tracks1, tracks2))
    matched_ids = matches[matches[:, 2] < 5][:, :2]
    np.savetxt(
        result_folder / args.save_name,
        matched_ids[:, :2],
        header="tracks1_ids, tracks2_ids",
        delimiter=",",
        fmt="%d",
    )

    # read stereo parameters and create 3D coordinates


if __name__ == "__main__":
    main()
