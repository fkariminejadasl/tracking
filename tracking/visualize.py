import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from tracking.data_association import (Detection, Point, get_detections,
                                       get_frame_numbers_of_track)


def get_video_parameters(vc: cv2.VideoCapture):
    if vc.isOpened():
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vc.get(cv2.CAP_PROP_FPS)
        return height, width, total_no_frames, fps
    else:
        return


def _create_output_video(
    output_video_file, vc, out_width=None, out_height=None, out_fps=None
):
    height, width, total_no_frames, fps = get_video_parameters(vc)
    if not out_width:
        out_width = width
    if not out_height:
        out_height = height
    if not out_fps:
        out_fps = fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"XVID") with .avi
    out = cv2.VideoWriter(
        output_video_file.as_posix(), fourcc, out_fps, (out_width, out_height)
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

    for frame_number in range(0, total_no_frames):
        frame = get_frame(frame_number, vc)

        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
        )
    out.release()


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

    for frame_number in range(0, total_no_frames):
        frame = get_frame(frame_number, vc)

        det_path = det_folder / f"{filename_fixpart}_{frame_number+1}.txt"
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


def plot_frameid_y(tracks, status, legned=False):
    _, ax = plt.subplots(1, 1)
    for k, track in tracks.items():
        frame_numbers = get_frame_numbers_of_track(track)
        if track.status == status:
            ax.plot(
                frame_numbers,
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
        track = tracks1[track_id1]
        frame_numbers = get_frame_numbers_of_track(track)
        ax1.plot(
            frame_numbers,
            [coords.y for coords in track.coords],
            "-*",
            color=track.color,
            label=f"track1: {track_id1}",
        )
    for track_id2 in track2_ids:
        track = tracks1[track_id2]
        frame_numbers = get_frame_numbers_of_track(track)
        ax1.plot(
            frame_numbers,
            [coords.y for coords in track.coords],
            "-*",
            color=track.color,
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
    total_no_frames: int = 0,
    fps: int = None,
    show_det_id=False,
    black=True,
):
    if total_no_frames != 0:
        out, height, width, _ = _create_output_video(
            output_video_file, vc, out_width, out_height, fps
        )
    else:
        out, height, width, total_no_frames = _create_output_video(
            output_video_file, vc, out_width, out_height, fps
        )

    for frame_number in range(0, total_no_frames):
        frame = get_frame(frame_number, vc)

        for track_id, track in tracks.items():
            frame_numbers = get_frame_numbers_of_track(track)
            if frame_number in frame_numbers:
                color = tuple(int(round(c * 255)) for c in track.color)
                color = (color[2], color[1], color[0])
                idx = frame_numbers.index(frame_number)
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
                if show_det_id:
                    text = f"{track_id},{coord.det_id}"
                else:
                    text = f"{track_id}"
                if black:
                    color = (0, 0, 0)
                cv2.putText(
                    frame,
                    text,
                    (int(coord.x) - w2, int(coord.y) - h2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    color,
                    1,  # Thinckness
                    2,  # line type
                )
        out = _write_frame_in_video(
            frame, out, frame_number, top_left, out_width, out_height
        )

    out.release()


def visualize_matches_in_video(
    all_matches,
    vc1,
    vc2,
    output_video_file,
    top_left1,
    bottom_right1,
    top_left2,
    bottom_right2,
    total_no_frames=0,
    fps: int = None,
):
    out_width = bottom_right1.x - top_left1.x + bottom_right2.x - top_left2.x
    out_height = bottom_right1.y - top_left1.y
    if total_no_frames != 0:
        out, height, width, _ = _create_output_video(
            output_video_file, vc1, out_width, out_height, fps
        )
    else:
        out, height, width, total_no_frames = _create_output_video(
            output_video_file, vc1, out_width, out_height, fps
        )
    font_scale = 1
    for frame_number in range(0, total_no_frames):
        frame1, frame2 = get_stereo_frames(frame_number, vc1, vc2)

        for track_id1, value in all_matches.items():
            for track_id2, matches in value.items():
                frame_numbers = [coord1.frame_number for coord1 in matches.coords1]
                if frame_number in frame_numbers:
                    color = tuple(int(round(c * 255)) for c in matches.track1_color)
                    color = (color[2], color[1], color[0])
                    idx = frame_numbers.index(frame_number)
                    coord = matches.coords1[idx]
                    w2 = int(coord.w / 2)
                    h2 = int(coord.h / 2)
                    text = f"{track_id1}"

                    cv2.rectangle(
                        frame1,
                        (int(coord.x) - w2, int(coord.y) - h2),
                        (int(coord.x) + w2, int(coord.y) + h2),
                        color=color,
                        thickness=1,
                    )
                    cv2.putText(
                        frame1,
                        text,
                        (int(coord.x) - w2, int(coord.y) - h2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,  # font scale
                        (0, 0, 0),
                        1,  # Thinckness
                        2,  # line type
                    )
                    cv2.putText(
                        frame1,
                        f"{frame_number}",
                        (top_left1.x + 15, top_left1.y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,  # font scale
                        (0, 0, 0),  # color
                        1,  # Thinckness
                        2,  # line type
                    )

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
                    cv2.putText(
                        frame2,
                        text,
                        (int(coord.x) - w2, int(coord.y) - h2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,  # font scale
                        (0, 0, 0),
                        1,  # Thinckness
                        2,  # line type
                    )
        frame1 = frame1[
            top_left1.y : bottom_right1.y,
            top_left1.x : bottom_right1.x,
            :,
        ]
        frame2 = frame2[
            top_left2.y : bottom_right2.y,
            top_left2.x : bottom_right2.x :,
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

    _, ax = plt.subplots(1, 1)
    _show_one_frame(ax, combined)


def _show_one_frame(ax, frame):
    ax.imshow(frame[..., ::-1])
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)


def _show_two_frames(axes, frame1, frame2):
    axes[0].imshow(frame1[..., ::-1])
    axes[1].imshow(frame2[..., ::-1])
    axes[0].axis("off")
    axes[1].axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


def _draw_detections_epipolar_lines(dets, ax, image_width, draw_text=True):
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


def _get_text_value(det: Detection, text: str) -> str:
    valid = ["det_id", "track_id", "frame_number"]
    assert text in valid, f"not valid. Use only: {valid}"
    if text == "det_id":
        text_value = str(f"{det.det_id}")
    if text == "track_id":
        text_value = str(f"{det.track_id}")
    if text == "frame_number":
        text_value = str(f"{det.frame_number}")
    return text_value


def _draw_detections(dets: list[Detection], ax, color="r", text="det_id"):
    for det in dets:
        ax.text(
            det.x - det.w // 2,
            det.y - det.h // 2,
            _get_text_value(det, text),
            color="r",
            fontsize=12,
        )
        rect = patches.Rectangle(
            (det.x - det.w // 2, det.y - det.h // 2),
            det.w,
            det.h,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def draw_detections_in_image(dets, image, color="r"):
    _, ax = plt.subplots(1, 1)
    _show_one_frame(ax, image)
    _draw_detections(dets, ax, color=color)


def show_detections_in_stereo(frame1, frame2, dets1, dets2, image_width):
    _, axes = plt.subplots(1, 2)
    _show_two_frames(axes, frame1, frame2)
    _draw_detections_epipolar_lines(dets1, axes[0], image_width)
    _draw_detections_epipolar_lines(dets2, axes[1], image_width)


def get_frame(frame_number, vc):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame = vc.read()
    return frame


def get_stereo_frames(frame_number, vc1, vc2):
    vc1.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    vc2.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame_c1 = vc1.read()
    _, frame_c2 = vc2.read()
    return frame_c1, frame_c2


def plot_two_tracks_stats(s1, s2):
    _, axs = plt.subplots(2, 3)
    axs[0, 0].plot(s1[:, 0], s1[:, 1], "-*", label="s1")
    axs[0, 0].plot(s1[:, 0], s2[:, 1], "-*", label="s2")
    axs[0, 1].plot(s1[:, 0], s1[:, 2], "-*")
    axs[0, 1].plot(s1[:, 0], s2[:, 2], "-*")
    axs[0, 2].plot(s1[:, 0], s1[:, 3], "-*")
    axs[0, 2].plot(s1[:, 0], s2[:, 3], "-*")
    axs[1, 0].plot(s1[:, 0], s1[:, 4], "-*")
    axs[1, 0].plot(s1[:, 0], s2[:, 4], "-*")
    axs[1, 1].plot(s1[:, 0], s1[:, 5], "-*")
    axs[1, 1].plot(s1[:, 0], s2[:, 5], "-*")
    axs[0, 0].legend()
    axs[0, 0].set_title("tp")
    axs[0, 1].set_title("fp")
    axs[0, 2].set_title("fn")
    axs[1, 0].set_title("sw")
    axs[1, 1].set_title("uid")
