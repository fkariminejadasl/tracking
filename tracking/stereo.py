from dataclasses import dataclass

from tracking.data_association import Detection, get_frame_numbers_of_track

accepted_track_length = 50
matched_track_length = 50
accepted_error = 3
smallest_disparity = 450
largest_disparity = 750


@dataclass
class Matches:
    error: float
    l1_norm: float
    count: int
    dets1: list[Detection]
    dets2: list[Detection]
    track1_color: tuple
    track2_color: tuple


def compute_possible_matches_for_track(track1, tracks2):
    possible_matches = {}
    for track_id2, track2 in tracks2.items():
        if len(track2.dets) > accepted_track_length:
            frame_numbers1 = get_frame_numbers_of_track(track1)
            frame_numbers2 = get_frame_numbers_of_track(track2)
            common_frames = set(frame_numbers1).intersection(set(frame_numbers2))
            error = 0
            count = 0
            dets1 = []
            dets2 = []
            if len(common_frames) > matched_track_length:
                for frame_id in common_frames:
                    det1 = [det for det in track1.dets if det.frame_number == frame_id][
                        0
                    ]
                    det2 = [det for det in track2.dets if det.frame_number == frame_id][
                        0
                    ]
                    l1_norm = abs(det1.y - det2.y)
                    if l1_norm < accepted_error:
                        error += l1_norm
                        count += 1
                        dets1.append(det1)
                        dets2.append(det2)
                if count != 0:
                    possible_matches[track_id2] = Matches(
                        error / count,
                        error,
                        count,
                        dets1,
                        dets2,
                        track1.color,
                        track2.color,
                    )
    matched_groups = {
        key: matches
        for key, matches in possible_matches.items()
        if len(matches.dets1) > matched_track_length
        if matches.error < 1
    }
    return matched_groups


def compute_possible_matches(tracks1, tracks2):
    all_matches = {}
    for track_id1, track1 in tracks1.items():
        if len(track1.dets) > accepted_track_length:
            matched_groups = compute_possible_matches_for_track(track1, tracks2)
            if matched_groups:
                all_matches[track_id1] = matched_groups
                print(f"{track_id1}: {list(matched_groups.keys())}")
    return all_matches


def save_matches(match_file, track_id1, matched_groups, cam_id):
    # inverse is true when first tracks2 and then tracks1
    with open(match_file, "a") as file:
        for track_id2, matches in matched_groups.items():
            for det1, det2 in zip(matches.dets1, matches.dets2):
                file.write(
                    f"{track_id1},{track_id2},{det1.frame_number},{cam_id},{det1.x},{det1.y},{det2.x},{det2.y}\n"
                )


def save_all_matches(match_file, all_matches, cam_id):
    for track_id1, matched_group in all_matches.items():
        save_matches(match_file, track_id1, all_matches[track_id1], cam_id)


@dataclass
class StereoItem:
    camera_id: int
    track_id: int
    det: Detection
    disp: int
    disp_prob: float


@dataclass
class Stereo:
    target: StereoItem
    candidates: list[StereoItem]


def compute_match_candidates(dets1, dets2, inverse=False) -> list[Stereo]:
    cam_id1 = 0
    cam_id2 = 1
    if inverse:
        cam_id1 = 1
        cam_id2 = 0
    matches = []
    for det1 in dets1:
        candidates = []
        for det2 in dets2:
            disp = det1.x - det2.x
            rectification_error = abs(det1.y - det2.y)
            if (
                rectification_error < accepted_error
                and disp < largest_disparity
                and disp > smallest_disparity
            ):
                st_item1 = StereoItem(cam_id1, -1, det1, disp, -1)
                st_item2 = StereoItem(cam_id2, -1, det2, disp, -1)
                candidates.append(st_item2)
        if candidates:
            match = Stereo(st_item1, candidates)
            matches.append(match)
    return matches
