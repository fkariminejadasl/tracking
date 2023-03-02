import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from sklearn.neighbors import KDTree
import numpy as np

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import tracking.data_association as da
import tracking.visualize as visualize


def _get_video_name_and_frame_number(image_path: Path) -> tuple[str, int]:
    split_name = image_path.stem.split("_frame_")
    video_name = split_name[0]
    frame_number = int(split_name[1])
    return video_name, frame_number


def get_other_image_path(path: Path, dtime: int = 1) -> Path | None:
    video_name, frame_number = _get_video_name_and_frame_number(path)
    if video_name not in [
        "04_07_22_F_2_rect_valid.mp4",
        "04_07_22_G_2_rect_valid.mp4",
    ]:  # for videos with 240 fps
        dtime = 8 * dtime
    next_image = path.parent / f"{video_name}_frame_{frame_number + dtime:06d}.jpg"
    if not next_image.exists():
        return
    return next_image


def get_crop_image(image_path, x_tl, y_tl, x_br, y_br):
    frame = cv2.imread(image_path.as_posix())
    c_frame = frame[y_tl:y_br, x_tl:x_br]
    return c_frame


def change_origin_bboxes(c_bboxes, x_tl, y_tl):
    c_bboxes[:, [3, 5, 7]] -= x_tl
    c_bboxes[:, [4, 6, 8]] -= y_tl
    return c_bboxes


def get_crop_bboxes(image_path, x_tl, y_tl, x_br, y_br, tracks):
    _, frame_number = _get_video_name_and_frame_number(image_path)

    image_bboxes = tracks[tracks[:, 1] == frame_number]
    c_bboxes = image_bboxes[
        (
            (image_bboxes[:, 7] >= x_tl)
            & (image_bboxes[:, 7] < x_br)
            & (image_bboxes[:, 8] >= y_tl)
            & (image_bboxes[:, 8] < y_br)
        )
    ]
    c_bboxes = change_origin_bboxes(c_bboxes, x_tl, y_tl)
    return c_bboxes


def zero_padding_bboxes(bboxes, number_bboxes):
    if bboxes.shape[0] < number_bboxes:
        pad_length = number_bboxes - bboxes.shape[0]
        bboxes = np.pad(bboxes, ((0, pad_length), (0, 0)))
    return bboxes


crop_width, crop_height = 512, 256
number_bboxes = 5
np.random.seed(342)
dtime = np.hstack((np.arange(-5,0), np.arange(1,5)))
dtime = 1 #np.random.permutation(dtime)[0]  # TODO iterate? -4 has issue
print(dtime)

# 1. random select image, d_time
image_path1 = Path(
    "/home/fatemeh/Downloads/data8_v1/train/images/183_cam_1_frame_002440.jpg"  # 406_cam_2_frame_001672.jpg"
)
video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)

# 2. image next: check(accept/reject) + select based on track
image_path2 = get_other_image_path(image_path1, dtime)
print(image_path1)
print(image_path2)

# 3. get bbox
tracks = da.load_tracks_from_mot_format(
    Path(f"/home/fatemeh/Downloads/vids/mot/{video_name}.zip")
)
bboxes1 = tracks[tracks[:, 1] == frame_number1]
#   3.1. multiple crops
track_ids = list(np.random.permutation(bboxes1[:, 0]))


# 4. bbox: check(accept/reject) + select
# frame1
"""
# multip boxes setup
c_bboxes1 = np.empty(shape=(0, bboxes1.shape[1]), dtype=np.int64)
while c_bboxes1.shape[0] < number_bboxes - 1:
    # if len(track_ids) == 0: # TODO next image?
    track_id1 = track_ids.pop()  # TODO iterate?
    bbox1 = bboxes1[bboxes1[:, 0] == track_id1][0]
    center_x, center_y = bbox1[7], bbox1[8]
    print(track_id1, center_x, center_y)
    x_tl, y_tl, x_br, y_br = da.tl_br_from_cen_wh(
        center_x, center_y, crop_width, crop_height
    )

    c_frame1 = get_crop_image(image_path1, x_tl, y_tl, x_br, y_br)
    c_bboxes1 = get_crop_bboxes(image_path1, x_tl, y_tl, x_br, y_br, tracks)

track_ids1 = c_bboxes1[:, 0]
c_bboxes1 = zero_padding_bboxes(c_bboxes1, number_bboxes)
"""

track_id1 = track_ids.pop()
bbox1 = bboxes1[bboxes1[:, 0] == track_id1][0]
center_x, center_y = bbox1[7], bbox1[8]
print(track_id1, center_x, center_y)
x_tl, y_tl, x_br, y_br = da.tl_br_from_cen_wh(
    center_x, center_y, crop_width, crop_height
)

c_frame1 = get_crop_image(image_path1, x_tl, y_tl, x_br, y_br)
c_bbox1 = change_origin_bboxes(bbox1.reshape((1, -1)), x_tl, y_tl)

visualize.plot_detections_in_image(
    da.make_dets_from_array(c_bbox1), c_frame1, "r", "track_id"
)

# frame2
c_frame2 = get_crop_image(image_path2, x_tl, y_tl, x_br, y_br)
c_bboxes2 = get_crop_bboxes(image_path2, x_tl, y_tl, x_br, y_br, tracks)
track_ids2 = c_bboxes2[:, 0]
print(c_bboxes2)

# adjust number of bboxes: for smaller one are zero padded, for larger ones knn used
if c_bboxes2.shape[0] < number_bboxes:
    c_bboxes2 = zero_padding_bboxes(c_bboxes2, number_bboxes)
if c_bboxes2.shape[0] > number_bboxes:
    kdt = KDTree(c_bboxes2[:, 7:9])
    ind = np.where(c_bboxes2[:, 0] == track_id1)[0]
    _, inds = kdt.query(c_bboxes2[ind, 7:9], k=number_bboxes)
    c_bboxes2 = c_bboxes2[inds[0]]

# make a label
inds = np.random.permutation(c_bboxes2.shape[0])
c_bboxes2 = c_bboxes2[inds]
c_bboxes2[:, 2] = np.arange(number_bboxes)
lable = c_bboxes2[c_bboxes2[:, 0] == track_id1, 2]
print(lable)
print(c_bboxes2)
visualize.plot_detections_in_image(
    da.make_dets_from_array(c_bboxes2), c_frame2, "r", "track_id"
)

"""
frame_number = 240
track_id = 21
time = 5
183_cam_1_240_21.jpg, txt
183_cam_1_245_21.jpg, txt

backbone(im1), backbone(im2), time, bbox1, bbox2
"""

