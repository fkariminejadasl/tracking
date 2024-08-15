from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from tracking.data_association import (
    Detection,
    Point,
    get_detections,
    get_frame_numbers_of_track,
    get_track_from_track_id,
    load_tracks_from_mot_format,
    tl_br_from_cen_wh,
)

# get_frame is too slow and replaced by vc.read().
# It seems due to vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number).


def save_image_with_dets(save_path, video_name, dets, image, format="06d"):
    """
    inputs:
        video_name: str
            just the name not extension
        dets: np.ndarray
            track_id, frame_number, det_id, xtl, ytl, xbr, ybr, xc, yc, w, h,
    """
    save_path.mkdir(parents=True, exist_ok=True)

    frame_number = dets[0, 1]
    color = (0, 0, 255)
    for det in dets:
        track_id = det[0]

        x_tl, y_tl, x_br, y_br = det[3:7]
        _put_bbox_in_image(image, x_tl, y_tl, x_br, y_br, color, track_id)

    cv2.putText(
        image,
        f"{frame_number}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # font scale
        (0, 0, 0),  # color
        1,  # Thinckness
        2,  # line type
    )

    name = f"{video_name}_frame_{frame_number:{format}}.jpg"
    cv2.imwrite((save_path / f"{name}").as_posix(), image)


def get_video_parameters(vc):
    clean = False
    if isinstance(vc, (Path, str)):
        clean = True
        vc = cv2.VideoCapture(str(vc))
    assert vc.isOpened()
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vc.get(cv2.CAP_PROP_FPS)
    if clean:
        vc.release()
    return height, width, total_no_frames, fps


def get_start_end_frames(start_frame, end_frame, total_no_frames):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_no_frames - 1
    return start_frame, end_frame


def split_video(vid_file, save_path):
    """
    Split a video into two halves, each capturing one half of the original video's width.

    Parameters
    ----------
    video_file : str|Path
        The path to the input video file.
    save_path : str|Path

    Returns
    -------
    None
        This function creates two output video files, one for each half of the original video.

    Notes
    -----
    The function reads an input video, splits each frame into left and right halves,
    and then writes these halves into two separate video files named 'name_1.mp4'
    and 'name_2.mp4'.
    """
    vid_file = Path(vid_file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vid_fix_name = vid_file.stem.split("_")[0]
    left_vid_file = save_path / f"{vid_fix_name}_1{vid_file.suffix}"
    right_vid_file = save_path / f"{vid_fix_name}_2{vid_file.suffix}"

    cap = cv2.VideoCapture(str(vid_file))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"XVID") with .avi

    left_writer = cv2.VideoWriter(str(left_vid_file), fourcc, fps, (width // 2, height))
    right_writer = cv2.VideoWriter(
        str(right_vid_file), fourcc, fps, (width // 2, height)
    )

    for frame_number in tqdm(range(0, total_no_frames)):
        _, frame = cap.read()

        left_half = frame[:, : width // 2]
        right_half = frame[:, width // 2 :]

        left_writer.write(left_half)
        right_writer.write(right_half)

    cap.release()
    left_writer.release()
    right_writer.release()


def _put_bbox_in_image(
    frame,
    x_tl,
    y_tl,
    x_br,
    y_br,
    color,
    track_id,
    show_det_id=False,
    det_id=0,
    black=True,
):
    cv2.rectangle(
        frame,
        (x_tl, y_tl),
        (x_br, y_br),
        color=color,
        thickness=1,
    )
    if show_det_id:
        text = f"{track_id},{det_id}"
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


def save_images_of_video(
    save_path: Path,
    video_file: Path,
    start_frame=None,
    end_frame=None,
    step: int = 1,
    format: str = "06d",
):
    save_path.mkdir(parents=True, exist_ok=True)

    vc = cv2.VideoCapture(video_file.as_posix())
    assert vc.isOpened()
    height, width, total_no_frames, fps = get_video_parameters(vc)
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )

    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            name = f"{video_file.stem}_frame_{frame_number:{format}}.jpg"
            cv2.imwrite((save_path / f"{name}").as_posix(), frame)
    vc.release()


def save_images_with_tracks(
    save_path: Path,
    video_file: Path,
    tracks: np.ndarray,
    start_frame=None,
    end_frame=None,
    step: int = 1,
    format: str = "06d",
):
    save_path.mkdir(parents=True, exist_ok=True)

    vc = cv2.VideoCapture(video_file.as_posix())
    assert vc.isOpened()
    height, width, total_no_frames, fps = get_video_parameters(vc)
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )

    tracks_ids = np.unique(tracks[:, 0])
    colors = np.random.randint(0, 255, size=(len(tracks_ids), 3))
    tracks_ids_to_inds = {track_id: i for i, track_id in enumerate(tracks_ids)}
    show_det_id = False
    black = True

    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            frame_tracks = tracks[tracks[:, 1] == frame_number]
            if frame_tracks.size == 0:
                continue
            for det in frame_tracks:
                det_id = det[2]
                track_id = det[0]
                color = colors[tracks_ids_to_inds[track_id]]
                color = tuple(map(int, color))
                color = (color[2], color[1], color[0])

                x_tl, y_tl, x_br, y_br = det[3:7]
                _put_bbox_in_image(
                    frame,
                    x_tl,
                    y_tl,
                    x_br,
                    y_br,
                    color,
                    track_id,
                    show_det_id,
                    det_id,
                    black,
                )

            cv2.putText(
                frame,
                f"{frame_number}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (0, 0, 0),  # color
                1,  # Thinckness
                2,  # line type
            )

            name = f"{video_file.stem}_frame_{frame_number:{format}}.jpg"
            cv2.imwrite((save_path / f"{name}").as_posix(), frame)

    vc.release()


def save_images_with_detections_mot(
    save_path: Path,
    video_file: Path,
    dets_path: Path,
    color=(0, 0, 255),
    start_frame=None,
    end_frame=None,
    step=1,
    format: str = "06d",
):
    """
    Yolo detections saved as {vid_name}_{frame_number + 1}.txt.
    Basically frame number is one-based. I use zeor-based.
    """
    all_dets = load_tracks_from_mot_format(dets_path)
    all_dets[:, 2] = all_dets[:, 0]
    all_dets[:, 0] = -1

    save_path.mkdir(parents=True, exist_ok=True)

    vc = cv2.VideoCapture(video_file.as_posix())
    assert vc.isOpened()
    height, width, total_no_frames, fps = get_video_parameters(vc)
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )

    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            dets = all_dets[all_dets[:, 1] == frame_number]

            for det in dets:
                x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(
                    det[7], det[8], det[9], det[10]
                )

                _put_bbox_in_image(
                    frame,
                    x_tl,
                    y_tl,
                    x_br,
                    y_br,
                    color,
                    det[2],
                )

            cv2.putText(
                frame,
                f"{frame_number}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (0, 0, 0),  # color
                1,  # Thinckness
                2,  # line type
            )

            name = f"{video_file.stem}_frame_{frame_number:{format}}.jpg"
            cv2.imwrite((save_path / f"{name}").as_posix(), frame)


def save_images_with_detections(
    save_path: Path,
    video_file: Path,
    dets_path: Path,
    color=(0, 0, 255),
    start_frame=None,
    end_frame=None,
    step=1,
    format: str = "06d",
):
    """
    Yolo detections saved as {vid_name}_{frame_number + 1}.txt.
    Basically frame number is one-based. I use zeor-based.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    vc = cv2.VideoCapture(video_file.as_posix())
    assert vc.isOpened()
    height, width, total_no_frames, fps = get_video_parameters(vc)
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )

    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            det_path = dets_path / f"{video_file.stem}_{frame_number+1}.txt"
            dets = get_detections(det_path, width, height, frame_number)

            for det in dets:
                x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)

                _put_bbox_in_image(
                    frame,
                    x_tl,
                    y_tl,
                    x_br,
                    y_br,
                    color,
                    det.det_id,
                )

            cv2.putText(
                frame,
                f"{frame_number}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (0, 0, 0),  # color
                1,  # Thinckness
                2,  # line type
            )

            name = f"{video_file.stem}_frame_{frame_number:{format}}.jpg"
            cv2.imwrite((save_path / f"{name}").as_posix(), frame)


def _create_output_video(
    output_video_file, vc, out_width=None, out_height=None, out_fps=None
):
    output_video_file.parent.mkdir(parents=True, exist_ok=True)

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

    return out, out_height, out_width, total_no_frames


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
    output_video_file,
    vc,
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
    start_frame=None,
    end_frame=None,
    step=1,
    fps=None,
):
    out, out_height, out_width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height, out_fps=fps
    )
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )
    vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()

        if frame_number % step == 0:
            out = _write_frame_in_video(
                frame, out, frame_number, top_left, out_width, out_height
            )
    out.release()


# TODO integrated in save_images_with_detections
def plot_detections_in_video(
    output_video_file: Path,
    video_file: Union[Path, cv2.VideoCapture],
    det_folder: Path,
    filename_fixpart: str,
    top_left=Point(0, 0),
    out_width=None,
    out_height=None,
    color=(0, 0, 255),
    start_frame=None,
    end_frame=None,
    step=1,
):
    if isinstance(video_file, cv2.VideoCapture):
        vc = video_file
    else:
        vc = cv2.VideoCapture(video_file.as_posix())

    height, width, _, _ = get_video_parameters(vc)

    out, out_height, out_width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height
    )
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            det_path = det_folder / f"{filename_fixpart}_{frame_number+1}.txt"
            dets = get_detections(det_path, width, height, frame_number)

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


# TODO integrated in save_images_with_tracks
def plot_tracks_array_in_video(
    output_video_file,
    tracks: np.ndarray,
    video_file,
    top_left=Point(0, 0),
    out_width=None,
    out_height=None,
    start_frame=None,
    end_frame=None,
    step=1,
    fps: int = None,
    show_det_id=False,
    black=False,
):
    vc = cv2.VideoCapture(video_file.as_posix())
    out, out_height, out_width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height, fps
    )

    tracks_ids = np.unique(tracks[:, 0])
    colors = np.random.randint(0, 255, size=(len(tracks_ids), 3))
    tracks_ids_to_inds = {track_id: i for i, track_id in enumerate(tracks_ids)}

    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            frame_tracks = tracks[tracks[:, 1] == frame_number]
            if frame_tracks.size == 0:
                continue
            for det in frame_tracks:
                det_id = det[2]
                track_id = det[0]
                color = colors[tracks_ids_to_inds[track_id]]
                color = tuple(map(int, color))
                color = (color[2], color[1], color[0])

                x_tl, y_tl, x_br, y_br = det[3:7]
                _put_bbox_in_image(
                    frame,
                    x_tl,
                    y_tl,
                    x_br,
                    y_br,
                    color,
                    track_id,
                    show_det_id,
                    det_id,
                    black,
                )

            out = _write_frame_in_video(
                frame, out, frame_number, top_left, out_width, out_height
            )

    out.release()
    vc.release()


# TODO integrated in save_images_with_tracks
def plot_tracks_in_video(
    output_video_file,
    tracks,
    vc,
    top_left=Point(1300, 700),
    out_width=900,
    out_height=500,
    start_frame=None,
    end_frame=None,
    step=1,
    fps: int = None,
    show_det_id=False,
    black=True,
):
    out, out_height, out_width, total_no_frames = _create_output_video(
        output_video_file, vc, out_width, out_height, fps
    )

    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        _, frame = vc.read()
        if frame_number % step == 0:
            for track_id, track in tracks.items():
                frame_numbers = get_frame_numbers_of_track(track)
                if frame_number in frame_numbers:
                    color = tuple(int(round(c * 255)) for c in track.color)
                    color = (color[2], color[1], color[0])
                    ind = frame_numbers.index(frame_number)
                    det = track.dets[ind]
                    x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(
                        det.x, det.y, det.w, det.h
                    )

                    det_id = det.det_id
                    _put_bbox_in_image(
                        frame,
                        x_tl,
                        y_tl,
                        x_br,
                        y_br,
                        color,
                        track_id,
                        show_det_id,
                        det_id,
                        black,
                    )

            out = _write_frame_in_video(
                frame, out, frame_number, top_left, out_width, out_height
            )

    out.release()


# TODO remove?
def plot_matches_in_video(
    all_matches,
    vc1,
    vc2,
    output_video_file,
    top_left1,
    bottom_right1,
    top_left2,
    bottom_right2,
    start_frame=None,
    end_frame=None,
    step=1,
    fps: int = None,
):
    font_scale = 1
    out_width = bottom_right1.x - top_left1.x + bottom_right2.x - top_left2.x
    out_height = bottom_right1.y - top_left1.y
    out, out_height, out_width, total_no_frames = _create_output_video(
        output_video_file, vc1, out_width, out_height, fps
    )
    start_frame, end_frame = get_start_end_frames(
        start_frame, end_frame, total_no_frames
    )
    for frame_number in tqdm(range(start_frame, end_frame + 1)):
        frame1, frame2 = get_stereo_frames(frame_number, vc1, vc2)
        if frame_number % step == 0:
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


# def _plot_detections_epipolar_lines(dets, ax, image_width, draw_text=True):
#     for det in dets:
#         ax.plot([0, image_width - 1], [det.y, det.y], "--r", linewidth=0.5, alpha=0.5)
#         x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
#         if draw_text:
#             ax.text(
#                 x_tl,
#                 y_tl,
#                 str(f"{det.det_id},{det.score:.2f}"),
#                 color="r",
#                 fontsize=12,
#             )
#         rect = patches.Rectangle(
#             (x_tl, y_tl),
#             det.w,
#             det.h,
#             linewidth=1,
#             edgecolor="r",
#             facecolor="none",
#         )
#         ax.add_patch(rect)


def _plot_detections_epipolar_lines(dets, ax, image_width, draw_text=True):
    for det in dets:
        # ax.plot([0, image_width - 1], [det.y, det.y], "--r", linewidth=0.5, alpha=0.5)
        # x_tl, y_tl, x_br, y_br = tl_br_from_cen_wh(det.x, det.y, det.w, det.h)
        ax.plot([0, image_width - 1], [det[8], det[8]], "--r", linewidth=0.5, alpha=0.5)
        x_tl, y_tl, x_br, y_br = det[3:7]
        if draw_text:
            ax.text(
                x_tl,
                y_tl,
                str(f"{det[0]},{det[-1]:.2f}"),
                color="r",
                fontsize=12,
            )
        rect = patches.Rectangle(
            (x_tl, y_tl),
            det[9],
            det[10],
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


def _plot_detections(dets: List[Detection], ax, color="r", text="det_id"):
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


def _plot_bboxs(bboxes, ax, color="r"):
    """bboxes: list of [id, x_tl, y_tl, x_br, y_br]"""
    for bbox in bboxes:
        text, x_tl, y_tl, x_br, y_br = bbox
        ax.text(
            x_tl,
            y_tl,
            text,
            color="r",
            fontsize=12,
        )
        rect = patches.Rectangle(
            (x_tl, y_tl),
            x_br - x_tl,
            y_br - y_tl,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_detections_in_image(dets, image, color="r", text="det_id"):
    """dets: is either Detection format or a list of [id, x_tl, y_tl, x_br, y_br]"""
    _, ax = plt.subplots(1, 1)
    show_one_frame(ax, image)
    if isinstance(dets[0], Detection):
        _plot_detections(dets, ax, color=color, text=text)
    else:
        _plot_bboxs(dets, ax, color="r")


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
