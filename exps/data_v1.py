import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking.visualize import get_frame, get_video_parameters


def save_video_frames_on_disk(
    save_path: Path,
    vc,
    start_frame: int,
    end_frame: int,
    step: int,
    name_prefix: str = "",
    format: str = "jpg",
):
    for frame_number in tqdm(range(start_frame, end_frame, step)):
        frame = get_frame(frame_number, vc)
        save_file = save_path / f"{name_prefix}frame_{frame_number:06d}.{format}"
        cv2.imwrite(save_file.as_posix(), frame)


def save_yolo_labels_on_disk(
    src_path: Path,
    dst_path: Path,
    start_frame: int,
    end_frame: int,
    step: int,
    name_prefix: str = "",
):
    for frame_number in tqdm(range(start_frame, end_frame, step)):
        label = f"frame_{frame_number:06d}.txt"
        src_file = src_path / label
        dst_file = dst_path / f"{name_prefix}{label}"
        shutil.copy2(src_file, dst_file)


def prepare_data_for_yolo_one_vid_tr_val(
    save_path: Path,
    video_path: Path,
    label_path: Path,
    video_name: str,
):
    start_frame, end_frame, step = 0, 64, 1  # 512, 8
    val_end_frame = end_frame + 32 * step
    name_prefix = Path(video_name).stem + "_"

    train_path = save_path / "train/images"
    valid_path = save_path / "valid/images"
    train_label_path = save_path / "train/labels"
    valid_label_path = save_path / "valid/labels"
    for path in [train_path, valid_path, train_label_path, valid_label_path]:
        path.mkdir(exist_ok=True, parents=True)

    # save images
    vc = cv2.VideoCapture((video_path / video_name).as_posix())
    save_video_frames_on_disk(train_path, vc, start_frame, end_frame, step, name_prefix)
    save_video_frames_on_disk(
        valid_path, vc, end_frame, val_end_frame, step, name_prefix
    )

    # save labels
    zip_file = label_path / (Path(video_name).stem + ".zip")
    shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
    yolo_path = zip_file.parent / zip_file.stem / "obj_train_data"
    save_yolo_labels_on_disk(
        yolo_path, train_label_path, start_frame, end_frame, step, name_prefix
    )
    save_yolo_labels_on_disk(
        yolo_path, valid_label_path, end_frame, val_end_frame, step, name_prefix
    )
    shutil.rmtree(yolo_path.parent)

    # save meta file
    # /home/fkarimineja/data
    with open(save_path / "data.yaml", "w") as wfile:
        location = Path("/home/fkarimineja/data") / save_path.stem
        remote_train_path = location / "train/images"
        remote_val_path = location / "valid/images"
        wfile.write(f"train: {remote_train_path}\n")
        wfile.write(f"val: {remote_val_path}\n\n")
        wfile.write("nc: 1\n")
        wfile.write("names: ['0']")


def prepare_data_for_yolo_one_vid(
    save_path: Path,
    video_path: Path,
    label_path: Path,
    video_name: str,  # with .mp4
    stage: str,  # train or val
    start_frame: int = 0,
    end_frame: int = None,
    step: int = 8,
):
    name_prefix = Path(video_name).stem + "_"

    train_image_path = save_path / f"{stage}/images"
    train_label_path = save_path / f"{stage}/labels"
    for path in [train_image_path, train_label_path]:
        path.mkdir(exist_ok=True, parents=True)

    # save images
    vc = cv2.VideoCapture((video_path / video_name).as_posix())
    height, width, total_no_frames, fps = get_video_parameters(vc)
    if end_frame is None:
        end_frame = total_no_frames
    save_video_frames_on_disk(
        train_image_path, vc, start_frame, end_frame, step, name_prefix
    )

    # save labels
    zip_file = label_path / (Path(video_name).stem + ".zip")
    print(zip_file)
    shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
    yolo_path = zip_file.parent / zip_file.stem / "obj_train_data"
    save_yolo_labels_on_disk(
        yolo_path, train_label_path, start_frame, end_frame, step, name_prefix
    )
    shutil.rmtree(yolo_path.parent)

    # save meta file
    with open(save_path / "data.yaml", "w") as wfile:
        location = Path("/home/fkarimineja/data") / save_path.stem
        remote_train_path = location / "train/images"
        remote_val_path = location / "valid/images"
        wfile.write(f"train: {remote_train_path}\n")
        wfile.write(f"val: {remote_val_path}\n\n")
        wfile.write("nc: 1\n")
        wfile.write("names: ['0']")


def prepare_data_for_yolo_all(
    save_path: Path,
    videos_main_path: Path,
    labels_main_path: Path,
):
    for video_path in videos_main_path.glob("*"):
        video_name = video_path.name
        if video_name == "231_cam_1.MP4":
            prepare_data_for_yolo_one_vid(
                save_path, videos_main_path, labels_main_path, video_name, "valid"
            )
        elif video_name in [
            "04_07_22_F_2_rect_valid.mp4",
            "04_07_22_G_2_rect_valid.mp4",
        ]:
            prepare_data_for_yolo_one_vid(
                save_path,
                videos_main_path,
                labels_main_path,
                video_name,
                "train",
                step=1,
            )
        else:
            prepare_data_for_yolo_one_vid(
                save_path, videos_main_path, labels_main_path, video_name, "train"
            )


if __name__ == "__main__":
    save_path = Path("~/Downloads/vids/dat8_v1/").expanduser()
    videos_main_path = Path("~/Downloads/vids/all").expanduser()
    labels_main_path = Path("~/Downloads/vids/yolo").expanduser()
    prepare_data_for_yolo_all(save_path, videos_main_path, labels_main_path)
