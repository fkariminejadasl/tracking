from dataclasses import dataclass

from data_association import Detection

accepted_track_length = 50
matched_track_length = 50
accepted_error = 3
smallest_disparity = 250
largest_disparity = 650


@dataclass
class Matches:
    error: float
    l1_norm: float
    count: int
    ids: list[int]
    coords1: list[Detection]
    coords2: list[Detection]
    track1_color: tuple
    track2_color: tuple


def compute_possible_matches_for_a_track(track1, tracks2):
    possible_matches = {}
    for track_id2, track2 in tracks2.items():
        if len(track2.coords) > accepted_track_length:
            frame_numbers1 = get_frame_numbers_of_track(track1)
            frame_numbers2 = get_frame_numbers_of_track(track2)
            common_frames = set(frame_numbers1).intersection(set(frame_numbers))
            error = 0
            count = 0
            ids = []
            coords1 = []
            coords2 = []
            if len(common_frames) > matched_track_length:
                for frame_id in common_frames:
                    coord1 = [
                        coord
                        for coord in track1.coords
                        if coord.frame_number == frame_id
                    ][0]
                    coord2 = [
                        coord
                        for coord in track2.coords
                        if coord.frame_number == frame_id
                    ][0]
                    l1_norm = abs(coord1.y - coord2.y)
                    if l1_norm < accepted_error:
                        error += l1_norm
                        count += 1
                        ids.append(frame_id)
                        coords1.append(coord1)
                        coords2.append(coord2)
                if count != 0:
                    possible_matches[track_id2] = Matches(
                        error / count,
                        error,
                        count,
                        ids,
                        coords1,
                        coords2,
                        track1.color,
                        track2.color,
                    )
    matched_groups = {
        key: matches
        for key, matches in possible_matches.items()
        if len(matches.ids) > matched_track_length
        if matches.error < 1
    }
    return matched_groups


def compute_possible_matches(tracks1, tracks2):
    all_matches = {}
    for track_id1, track1 in tracks1.items():
        if len(track1.coords) > accepted_track_length:
            matched_groups = compute_possible_matches_for_a_track(track1, tracks2)
            if matched_groups:
                all_matches[track_id1] = matched_groups
                print(f"{track_id1}: {list(matched_groups.keys())}")
    return all_matches


def save_matches(match_file, track_id1, matched_groups, inverse=False):
    # inverse is true when first tracks2 and then tracks1
    with open(match_file, "a") as file:
        for track_id2, matches in matched_groups.items():
            for coord1, coord2 in zip(matches.coords1, matches.coords2):
                file.write(
                    f"{track_id1},{track_id2},{coord1.x},{coord1.y},{coord1.frame_number},{coord2.x},{coord2.y},{coord2.frame_number},{int(inverse)}\n"
                )


def save_all_matches(match_file, all_matches, inverse=False):
    for track_id1, matched_group in all_matches.items():
        save_matches(match_file, track_id1, all_matches[track_id1], inverse)


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
