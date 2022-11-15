import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from data_association import Point, get_detections, get_video_parameters


def _create_output_video(output_video_file, vc, out_width=None, out_height=None):
    height, width, total_no_frames, fps = get_video_parameters(vc)
    if not out_width:
        out_width = width
    if not out_height:
        out_height = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"XVID") with .avi
    out = cv2.VideoWriter(
        output_video_file.as_posix(), fourcc, fps, (out_width, out_height)
    )

    return out, height, width, total_no_frames


def _write_frame_in_video(frame, out, frame_number, top_left, out_width, out_height):
    cv2.putText(
        frame,
        f"{frame_number}",
        (top_left.x + 15, top_left.y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # font scale
        (0, 0, 0),  # color
        1,  # Thinckness
        2,  # line type
    )
    out.write(
        frame[
            int(top_left.y) : int(top_left.y) + out_height,
            int(top_left.x) : int(top_left.x) + out_width,
            :,
        ]
    )

    return out


def crop_video(
    vc,
    output_video_file,
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    out, height, width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height
    )

    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
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
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
    color=(0, 0, 255),
):
    out, height, width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height
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
        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
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
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    out, height, width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height
    )

    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        for track_id, track in tracks.items():
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
                cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(coord.x) - w2, int(coord.y) - h2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (0, 0, 0),  # color,
                    1,  # Thinckness
                    2,  # line type
                )
                # show as thick points
                # for i in range(6):
                #     for j in range(6):
                #         frame[coord.y + i, coord.x + j, :] = color
        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
        )

    out.release()


def visualize_tracks_in_video_from_file(
    track_file,
    vc,
    output_video_file,
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
):
    if isinstance(tracks, list):
        tracks = np.array(track_file)
    if isinstance(tracks, Path):
        tracks = np.array(read_tracks(track_file))

    out, height, width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height
    )

    for frame_number in range(1, total_no_frames + 1):
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = vc.read()

        indices = np.where(tracks[:, 1] == frame_number)[0]
        # color = np.random.randint(0, 255, size=(3))
        # color = tuple(int(c) for c in color)
        color = (236, 186, 220)
        for idx in indices:
            track = tracks[idx]
            track_id = int(track[0])

            x = int(track[3])
            y = int(track[4])
            w2 = int(track[5] / 2)
            h2 = int(track[6] / 2)
            cv2.rectangle(
                frame,
                (x - w2, y - h2),
                (x + w2, y + h2),
                color=color,
                thickness=1,
            )
            cv2.putText(
                frame,
                f"{track_id}",
                (x - w2, y - h2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (0, 0, 0),  # color,
                1,  # Thinckness
                2,  # line type
            )
            # show as thick points
            # for i in range(6):
            #     for j in range(6):
            #         frame[coord.y + i, coord.x + j, :] = color
        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
        )

    out.release()


def visualize_matches_in_video(
    all_matches,
    vc1,
    vc2,
    output_video_file,
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
    inverse=False,
):

    out, height, width, total_no_frames = _create_output_video(
        output_video_file, vc1, out_width * 2, out_height
    )

    for frame_number in range(1, total_no_frames + 1):
        vc1.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame1 = vc1.read()
        vc2.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame2 = vc2.read()

        for track_id1, value in all_matches.items():
            for track_id2, matches in value.items():
                frameids = [coord1.frame_number for coord1 in matches.coords1]
                if frame_number in frameids:
                    color = tuple(int(round(c * 255)) for c in matches.track1_color)
                    color = (color[2], color[1], color[0])
                    idx = np.where(np.array(frameids) == frame_number)[0][0]
                    coord = matches.coords1[idx]
                    w2 = int(coord.w / 2)
                    h2 = int(coord.h / 2)
                    if inverse:
                        cv2.rectangle(
                            frame2,
                            (int(coord.x) - w2, int(coord.y) - h2),
                            (int(coord.x) + w2, int(coord.y) + h2),
                            color=color,
                            thickness=1,
                        )
                    else:
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
                    if inverse:
                        cv2.rectangle(
                            frame1,
                            (int(coord.x) - w2, int(coord.y) - h2),
                            (int(coord.x) + w2, int(coord.y) + h2),
                            color=color,
                            thickness=1,
                        )
                    else:
                        cv2.rectangle(
                            frame2,
                            (int(coord.x) - w2, int(coord.y) - h2),
                            (int(coord.x) + w2, int(coord.y) + h2),
                            color=color,
                            thickness=1,
                        )
        frame1 = frame1[
            int(top_left.y) : int(top_left.y) + out_height,
            int(top_left.x) : int(top_left.x) + out_width,
            :,
        ]
        frame2 = frame2[
            int(top_left.y) : int(top_left.y) + out_height,
            int(top_left.x - 200) : int(top_left.x - 200) + out_width,
            :,
        ]
        out.write(np.concatenate((frame1, frame2), axis=1))
    out.release()


def superimpose_two_images(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(frame1)
    combined[..., 0] = gray1
    combined[..., 1] = gray2
    combined[..., 2] = gray2
    plt.figure()
    plt.imshow(combined)
    plt.show(block=False)


def _draw_detections_on_image(dets, ax, image_width, draw_text=True):
    for det in dets:
        ax.plot([0, image_width - 1], [det.y, det.y], "--r", linewidth=0.5, alpha=0.5)
        if draw_text:
            ax.text(
                det.x - det.w // 2,
                det.y - det.h // 2,
                str(f"{det.det_id},{det.score:.2f}"),
                color="r",
                fontsize=12,
            )
        rect = patches.Rectangle(
            (det.x - det.w // 2, det.y - det.h // 2),
            det.w,
            det.h,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)


def draw_detections_in_stereo(frame1, frame2, dets1, dets2, image_width):
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(frame1[..., ::-1])
    axes[1].imshow(frame2[..., ::-1])
    axes[0].axis("off")
    axes[1].axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    _draw_detections_on_image(dets1, axes[0], image_width)
    _draw_detections_on_image(dets2, axes[1], image_width)


"""
fig, axs = plt.subplots(1,3)
for track_id in [21,19,20,23]:
    track = tracks1[track_id] # track = tracks1[21]
    axs[0].plot(track.frameids, [coord.x for coord in track.coords], '*-', color=track.color, label=str(track.predicted_loc.det_id))
    axs[1].plot(track.frameids, [coord.y for coord in track.coords], '*-', color=track.color, label=str(track.predicted_loc.det_id))
    axs[2].plot([coord.x for coord in track.coords], [coord.y for coord in track.coords], '*-', color=track.color, label=str(track.predicted_loc.det_id))
axs[0].set_xlabel('frame');axs[0].set_ylabel('x')
axs[1].set_xlabel('frame');axs[1].set_ylabel('y')
axs[2].set_xlabel('x');axs[2].set_ylabel('y')
axs[0].legend();axs[1].legend();axs[2].legend()
plt.show(block=False)

from mpl_toolkits import mplot3d
plt.figure()
ax = plt.axes(projection='3d')
for track_id in [21,19,20,23]:
    track = tracks1[track_id]
    ax.plot([coord.x for coord in track.coords], [coord.y for coord in track.coords], track.frameids, '*-', color=track.color, label=str(track.predicted_loc.det_id))
ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('frame')
ax.legend()
plt.show(block=False)

def plot_ious(tracks, track_ids):
    plt.figure()
    for track_id in track_ids:
        track = tracks[track_id]
        ious = []
        for frame_id, det1, det2 in zip(track.frameids[1:], track.coords[1:], track.coords[0:-1]):
            iou = get_iou(det1,det2)
            ious.append(get_iou(det1,det2))
            # print(f"{frame_id}, {iou:.2f}")
        idx = np.where(np.diff(track.frameids) !=1)[0]
        print(f"{track_id}: {np.diff(track.frameids)[idx]}")

        plt.plot(track.frameids[1:], ious,  '*-', color=track.color, label=str(track_id))
        if idx.size>0:
            plt.plot([np.array(track.frameids)[idx]], [np.array(ious)[idx]],'or',markerfacecolor='none')
    plt.legend()
    plt.show(block=False)
"""
