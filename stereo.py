from dataclasses import dataclass

from data_association import Detection

accepted_track_length = 50
matched_track_length = 50
accepted_error = 3
# match_file = result_folder/"matches.txt"


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
        if len(track2.frameids) > accepted_track_length:
            # TODO: [1:] is a hack for the existing bug
            common_frames = set(track1.frameids[1:]).intersection(
                set(track2.frameids[1:])
            )
            error = 0
            count = 0
            ids = []
            coords1 = []
            coords2 = []
            if len(common_frames) > matched_track_length:
                for frame_id in common_frames:
                    coord1 = [coord for coord in track1.coords if coord.id == frame_id][
                        0
                    ]
                    coord2 = [coord for coord in track2.coords if coord.id == frame_id][
                        0
                    ]
                    l1_norm = abs(coord1.y - coord2.y)
                    if l1_norm < accepted_error:
                        error += l1_norm
                        count += 1
                        ids.append(frame_id)
                        coords1.append(coord1)
                        coords2.append(coord2)
                if count != 0:
                    possible_matches[track_id2] = Matches(
                        error / count, error, count, ids, coords1, coords2, track1.color, track2.color
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
        if len(track1.frameids) > accepted_track_length:
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
                    f"{track_id1},{track_id2},{coord1.x},{coord1.y},{coord1.id},{coord2.x},{coord2.y},{coord2.id},{int(inverse)}\n"
                )


def save_all_matches(match_file, all_matches, inverse=False):
    for track_id1, matched_group in all_matches.items():
        save_matches(match_file, track_id1, all_matches[track_id1], inverse)
