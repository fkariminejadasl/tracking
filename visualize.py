import cv2
import numpy as np
from matplotlib import pyplot as plt

from data_association import Point, get_detections, get_video_parameters


def crop_video(
    vc,
    output_video_file,
    focus_point=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    _, _, total_no_frames, fps = get_video_parameters(vc)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_file.as_posix(), fourcc, fps, (out_width, out_height)
    )

    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        out.write(
            frame[
                int(focus_point.y) : int(focus_point.y) + out_height,
                int(focus_point.x) : int(focus_point.x) + out_width,
                :,
            ]
        )
    out.release()

def draw_detections_in_a_frame(frame, dets):
    for det in dets:
        w2 = int(det.w / 2)
        h2 = int(det.h / 2)
        color = (0, 0, 255)
        cv2.rectangle(
            frame,
            (int(det.x) - w2, int(det.y) - h2),
            (int(det.x) + w2, int(det.y) + h2),
            color=color,
            thickness=1,
        )
    plt.figure()
    plt.imshow(frame[..., ::-1])
    plt.show(block=False)


def visualize_detections_in_video(
    filename_fixpart,
    det_folder,
    vc,
    output_video_file,
    focus_point=Point(1300, 700),
    out_width=900,
    out_height=500,
    color=(0, 0, 255),
):
    height, width, total_no_frames, fps = get_video_parameters(vc)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_file.as_posix(), fourcc, fps, (out_width, out_height)
    )

    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        det_path = det_folder / f"{filename_fixpart}_{frame_number}.txt"
        dets = get_detections(det_path, frame_number, width, height)

        for coord in dets:
            w2 = int(coord.w / 2)
            h2 = int(coord.h / 2)
            cv2.rectangle(
                frame,
                (int(coord.x) - w2, int(coord.y) - h2),
                (int(coord.x) + w2, int(coord.y) + h2),
                color=color,
                thickness=1,
            )
            # show as thick points
            # for i in range(6):
            #     for j in range(6):
            #         frame[int(coord.y) + i, int(coord.x) + j, :] = np.array(color)
        out.write(
            frame[
                int(focus_point.y) : int(focus_point.y) + out_height,
                int(focus_point.x) : int(focus_point.x) + out_width,
                :,
            ]
        )
    out.release()


def draw_matches(frame1, frame2, matches1, matches2):
    _, ax1 = plt.subplots(1, 1)
    _, ax2 = plt.subplots(1, 1)
    ax1.imshow(frame1[..., ::-1])
    ax2.imshow(frame2[..., ::-1])
    ax1.set_xlim(1300, 2200)
    ax1.set_ylim(1200, 700)
    ax2.set_xlim(1100, 2000)
    ax2.set_ylim(1200, 700)

    for match1, match2 in zip(matches1, matches2):
        ax1.plot([match1.x, match2.x], [match1.y, match2.y], "*-", color=(0, 0, 1))
        ax2.plot([match1.x, match2.x], [match1.y, match2.y], "*-", color=(0, 0, 1))
    plt.show(block=False)


def visualize_tracks_on_a_frame(tracks, vc, frame_number=1):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    _, frame1 = vc.read()
    plt.figure()
    plt.imshow(frame1[..., ::-1])
    for _, track in tracks.items():
        plt.plot(
            [det.x for det in track.coords],
            [det.y for det in track.coords],
            "*-",
            color=track.color,
        )
    plt.show(block=False)


def visualize_tracks_on_two_frames(tracks, vc, frame_number1, frame_number2):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number1 - 1)
    _, frame1 = vc.read()
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number2 - 1)
    _, frame2 = vc.read()
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    ax1.imshow(frame1[..., ::-1])
    ax2.imshow(frame2[..., ::-1])
    for _, track in tracks.items():
        for id in track.frameids:
            if id == frame_number1:
                ax1.plot(
                    [track.coords[0].x], [track.coords[0].y], "*", color=track.color
                )
            if id == frame_number2:
                ax2.plot(
                    [track.coords[-1].x], [track.coords[-1].y], "*", color=track.color
                )
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.show(block=False)


# roughly up to 50 frames they can be tracked
# c = 0
# plt.figure()
# for k, track in tracks.items():
#     if track.status == Status.Tracked:
#         c += 1
#         print(f"{k},{len(track.frameids)}")
#         plt.plot(track.frameids,"*-",label=str(k))
# print(f"{c}")


def plot_frameid_y(tracks, status, legned=False):
    _, ax = plt.subplots(1, 1)
    for k, track in tracks.items():
        if track.status == status:
            ax.plot(
                track.frameids,
                [coord.y for coord in track.coords],
                "*-",
                color=track.color,
                label=str(k),
            )
            print(f"{k}, {track.status}")
    if legned == True:
        ax.legend()
    plt.show(block=False)


def plot_frameid_y_for_stereo(tracks1, track1_ids, tracks2, track2_ids):
    track_id1 = 15
    _, ax1 = plt.subplots(1, 1)
    for track_id1 in track1_ids:
        ax1.plot(
            tracks1[track_id1].frameids,
            [coords.y for coords in tracks1[track_id1].coords],
            "-*",
            color=tracks1[track_id1].color,
            label=f"track1: {track_id1}",
        )
    for track_id2 in track2_ids:
        ax1.plot(
            tracks2[track_id2].frameids,
            [coords.y for coords in tracks2[track_id2].coords],
            "-*",
            color=tracks2[track_id2].color,
            label=f"track2: {track_id2}",
        )
    ax1.legend()
    plt.show(block=False)


def visualize_tracks_in_video(
    tracks,
    vc,
    output_video_file,
    focus_point=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    _, _, total_no_frames, fps = get_video_parameters(vc)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_file.as_posix(),
        fourcc,
        fps,
        (out_width, out_height),  # (width, height)
    )
    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        for _, track in tracks.items():
            if frame_number in track.frameids:
                color = tuple(int(round(c * 255)) for c in track.color)
                color = (color[2], color[1], color[0])
                idx = np.where(np.array(track.frameids) == frame_number)[0][0]
                coord = track.coords[idx]
                w2 = int(coord.w / 2)
                h2 = int(coord.h / 2)
                cv2.rectangle(
                    frame,
                    (int(coord.x) - w2, int(coord.y) - h2),
                    (int(coord.x) + w2, int(coord.y) + h2),
                    color=color,
                    thickness=1,
                )
                # show as thick points
                # for i in range(6):
                #     for j in range(6):
                #         frame[coord.y + i, coord.x + j, :] = color
        out.write(
            frame[
                int(focus_point.y) : int(focus_point.y) + out_height,
                int(focus_point.x) : int(focus_point.x) + out_width,
                :,
            ]
        )  # cam1: 1300:2200; cam2: 1100:2000, height: 700:1200
    out.release()


def visualize_matches_in_video(
    all_matches,
    vc1,
    vc2,
    output_video_file,
    focus_point=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    _, _, total_no_frames, fps = get_video_parameters(vc1)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_file.as_posix(),
        fourcc,
        fps,
        (out_width * 2, out_height),  # (width, height)
    )
    for frame_number in range(1, total_no_frames + 1):
        vc1.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame1 = vc1.read()
        vc2.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame2 = vc2.read()

        for track_id1, value in all_matches.items():
            for track_id2, matches in value.items():
                frameids = [coord1.id for coord1 in matches.coords1]
                if frame_number in frameids:
                    color = tuple(int(round(c * 255)) for c in matches.track1_color)
                    color = (color[2], color[1], color[0])
                    idx = np.where(np.array(frameids) == frame_number)[0][0]
                    coord = matches.coords1[idx]
                    w2 = int(coord.w / 2)
                    h2 = int(coord.h / 2)
                    cv2.rectangle(
                        frame1,
                        (int(coord.x) - w2, int(coord.y) - h2),
                        (int(coord.x) + w2, int(coord.y) + h2),
                        color=color,
                        thickness=1,
                    )
                    color = tuple(int(round(c * 255)) for c in matches.track2_color)
                    color = (color[2], color[1], color[0])
                    coord = matches.coords2[idx]
                    w2 = int(coord.w / 2)
                    h2 = int(coord.h / 2)
                    cv2.rectangle(
                        frame2,
                        (int(coord.x) - w2, int(coord.y) - h2),
                        (int(coord.x) + w2, int(coord.y) + h2),
                        color=color,
                        thickness=1,
                    )
        frame1 = frame1[
            int(focus_point.y) : int(focus_point.y) + out_height,
            int(focus_point.x) : int(focus_point.x) + out_width,
            :,
        ]
        frame2 = frame2[
            int(focus_point.y) : int(focus_point.y) + out_height,
            int(focus_point.x - 200) : int(focus_point.x - 200) + out_width,
            :,
        ]
        out.write(
            np.concatenate((frame1, frame2), axis=1)
        )  # cam1: 1300:2200; cam2: 1100:2000, height: 700:1200
    out.release()
