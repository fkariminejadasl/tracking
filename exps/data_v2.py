import shutil
from pathlib import Path


def create_folders(dst_path: Path):
    for stage in ["train", "valid"]:
        for type_ in ["labels", "images"]:
            new_path = dst_path / f"{stage}/{type_}"
            new_path.mkdir(parents=True, exist_ok=True)


def _copy_image(image_file: Path, dst_path: Path, stage: str):
    dst_file = dst_path / f"{stage}/images/{image_file.name}"
    shutil.copy2(image_file, dst_file)


def _copy_label(image_file: Path, dst_path: Path, stage: str):
    label_part = f"labels/{image_file.stem}.txt"
    label_file = image_file.parents[1] / label_part
    dst_file = dst_path / f"{stage}/{label_part}"
    shutil.copy2(label_file, dst_file)


def separate_train_valid(
    src_path: Path,
    dst_path: Path,
):
    image_files = list(src_path.glob("images/*jpg"))
    for i in range(94):
        _copy_image(image_files[i], dst_path, "valid")
        _copy_label(image_files[i], dst_path, "valid")

    for i in range(94, len(image_files)):
        _copy_image(image_files[i], dst_path, "train")
        _copy_label(image_files[i], dst_path, "train")


def save_meta_file(dst_path: Path):
    # save meta file
    with open(dst_path / "data.yaml", "w") as wfile:
        location = Path("/home/fkarimineja/data") / dst_path.stem
        remote_train_path = location / "train/images"
        remote_val_path = location / "valid/images"
        wfile.write(f"train: {remote_train_path}\n")
        wfile.write(f"val: {remote_val_path}\n\n")
        wfile.write("nc: 1\n")
        wfile.write("names: ['0']")


if __name__ == "__main__":
    src_path = (
        Path("~/Downloads/combChromis_norm.v5i.yolov5pytorch/train")
        .expanduser()
        .resolve()
    )
    dst_path = Path("~/Downloads/data8_v2").expanduser().resolve()

    create_folders(dst_path)
    separate_train_valid(src_path, dst_path)
    save_meta_file(dst_path)

"""
ln -s ~/data/data8_v1/valid/labels/* ~/data/data8_v2/valid/labels
ln -s ~/data/data8_v1/valid/images/* ~/data/data8_v2/valid/images
ln -s ~/data/data8_v1/train/labels/* ~/data/data8_v2/train/labels
ln -s ~/data/data8_v1/train/images/* ~/data/data8_v2/train/images
"""
