import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from tracking.data_association import (
    Detection,
    Point,
    get_a_track_from_track_id,
    get_detections,
    get_frame_numbers_of_track,
    tl_br_from_cen_wh,
)
from tracking.stereo_gt import get_disparity_info_from_stereo_track


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


def show_cropped_video(
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


def plot_detections_in_video(
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

        for det in dets:
            x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
            cv2.rectangle(
                frame,
                (x_tl, y_tl),
                (x_br, y_br),
                color=color,
                thickness=1,
            )
            # show as thick points
            # for i in range(6):
            #     for j in range(6):
            #         frame[int(det.y) + i, int(det.x) + j, :] = np.array(color)
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
                [det.y for det in track.dets],
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
            [det.y for det in track.dets],
            "-*",
            color=track.color,
            label=f"track1: {track_id1}",
        )
    for track_id2 in track2_ids:
        track = tracks1[track_id2]
        frame_numbers = get_frame_numbers_of_track(track)
        ax1.plot(
            frame_numbers,
            [det.y for det in track.dets],
            "-*",
            color=track.color,
            label=f"track2: {track_id2}",
        )
    ax1.legend()
    plt.show(block=False)


def plot_tracks_in_video(
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
                ind = frame_numbers.index(frame_number)
                det = track.dets[ind]
                x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
                cv2.rectangle(
                    frame,
                    (x_tl, y_tl),
                    (x_br, y_br),
                    color=color,
                    thickness=1,
                )
                if show_det_id:
                    text = f"{track_id},{det.det_id}"
                else:
                    text = f"{track_id}"
                if black:
                    color = (0, 0, 0)
                cv2.putText(
                    frame,
                    text,
                    (x_tl, y_tl),
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


def plot_matches_in_video(
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
                frame_numbers = [det1.frame_number for det1 in matches.dets1]
                if frame_number in frame_numbers:
                    color = tuple(int(round(c * 255)) for c in matches.track1_color)
                    color = (color[2], color[1], color[0])
                    ind = frame_numbers.index(frame_number)
                    det = matches.dets1[ind]
                    text = f"{track_id1}"

                    x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(
                        det.x, det.y, det.w, det.h
                    )
                    cv2.rectangle(
                        frame1,
                        (x_tl, y_tl),
                        (x_br, y_br),
                        color=color,
                        thickness=1,
                    )
                    cv2.putText(
                        frame1,
                        text,
                        (x_tl, y_tl),
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

                    det = matches.dets2[ind]
                    x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(
                        det.x, det.y, det.w, det.h
                    )
                    cv2.rectangle(
                        frame2,
                        (x_tl, y_tl),
                        (x_br, y_br),
                        color=color,
                        thickness=1,
                    )
                    cv2.putText(
                        frame2,
                        text,
                        (x_tl, y_tl),
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


def show_superimposed_two_images(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(frame1)
    combined[..., 0] = gray1
    combined[..., 1] = gray2
    combined[..., 2] = gray2

    _, ax = plt.subplots(1, 1)
    show_one_frame(ax, combined)


def show_one_frame(ax, frame):
    ax.imshow(frame[..., ::-1])
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)


def show_two_frames(axes, frame1, frame2):
    axes[0].imshow(frame1[..., ::-1])
    axes[1].imshow(frame2[..., ::-1])
    axes[0].axis("off")
    axes[1].axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


def _plot_detections_epipolar_lines(dets, ax, image_width, draw_text=True):
    for det in dets:
        ax.plot([0, image_width - 1], [det.y, det.y], "--r", linewidth=0.5, alpha=0.5)
        x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
        if draw_text:
            ax.text(
                x_tl,
                y_tl,
                str(f"{det.det_id},{det.score:.2f}"),
                color="r",
                fontsize=12,
            )
        rect = patches.Rectangle(
            (x_tl, y_tl),
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


def _plot_detections(dets: list[Detection], ax, color="r", text="det_id"):
    for det in dets:
        x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
        ax.text(
            x_tl,
            y_tl,
            _get_text_value(det, text),
            color="r",
            fontsize=12,
        )
        rect = patches.Rectangle(
            (x_tl, y_tl),
            det.w,
            det.h,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_detections_in_image(dets, image, color="r", text="det_id"):
    _, ax = plt.subplots(1, 1)
    show_one_frame(ax, image)
    _plot_detections(dets, ax, color=color, text=text)


def plot_detections_in_stereo(frame1, frame2, dets1, dets2, image_width):
    _, axes = plt.subplots(1, 2)
    show_two_frames(axes, frame1, frame2)
    _plot_detections_epipolar_lines(dets1, axes[0], image_width)
    _plot_detections_epipolar_lines(dets2, axes[1], image_width)


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


# TOBE removed
def plot_pred_dets_in_four_plots(frame1, frame2, dets1, dets2, pred_dets):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    [show_one_frame(axs[i, j], frame2) for i in range(2) for j in range(2)]
    _plot_detections(pred_dets, axs[1, 0], "r")
    _plot_detections(pred_dets, axs[0, 0], "r", "track_id")
    _plot_detections(pred_dets, axs[0, 1], "r", "frame_number")
    # _plot_detections(dets1, axs[0, 1], "g")
    _plot_detections(dets2, axs[1, 1], "r")


def plot_disparity_info(disparity_info: np.ndarray, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2)
    assert len(axs) >= 2
    label = f"{disparity_info[0,0]},{disparity_info[0,1]}"
    axs[0].plot(disparity_info[:, 2], disparity_info[:, 3], label=label)
    axs[1].plot(disparity_info[:, 2], disparity_info[:, 4], label=label)
    axs[0].legend()
    axs[1].legend()


def plot_disparity_infos(tracks1, tracks2, matches):
    for i, (track1_id, track2_id) in enumerate(matches):
        track1 = get_a_track_from_track_id(tracks1, track1_id)
        track2 = get_a_track_from_track_id(tracks2, track2_id)
        disparity_info = np.array(get_disparity_info_from_stereo_track(track1, track2))
        print(i, i % 10, track1_id, track2_id, len(disparity_info))
        if i % 10 == 0:
            fig, axs = plt.subplots(1, 2)
        plot_disparity_info(disparity_info, axs)
