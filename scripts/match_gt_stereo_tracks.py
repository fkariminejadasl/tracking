from pathlib import Path

import tracking.data_association as da
from tracking.stereo_track import (
    get_stereo_parameters,
    match_gt_stereo_tracks,
    rectify_tracks,
)
from tracking.utils_general import parse_args


def main(inputs):
    max_dist = 100

    save_file = Path(inputs.save_file)
    track_file1 = Path(inputs.track_file1)
    track_file2 = Path(inputs.track_file2)
    mat_file = Path(inputs.mat_file)
    im_width = inputs.image_width
    im_height = inputs.image_height

    (
        cameraMatrix1,
        distCoeffs1,
        R1,
        r_P1,
        cameraMatrix2,
        distCoeffs2,
        R2,
        r_P2,
    ) = get_stereo_parameters(mat_file, im_width, im_height)
    otracks1 = da.load_tracks_from_mot_format(track_file1)
    otracks2 = da.load_tracks_from_mot_format(track_file2)
    tracks1 = rectify_tracks(otracks1, cameraMatrix1, distCoeffs1, R1, r_P1)
    tracks2 = rectify_tracks(otracks2, cameraMatrix2, distCoeffs2, R2, r_P2)
    g1g2_tids, g1g2_frames_tids = match_gt_stereo_tracks(tracks1, tracks2, max_dist)
    print(g1g2_tids)
    print(g1g2_frames_tids)

    with open(save_file, "w") as file:
        # Write g1g2_tids
        file.write("g1g2_tids:\n")
        file.write(str(g1g2_tids) + "\n")

        # Write g1g2_frames_tids
        file.write("g1g2_frames_tids:\n")
        for frame in g1g2_frames_tids:
            file.write(str(frame) + "\n")


if __name__ == "__main__":
    inputs = parse_args()
    main(inputs)
