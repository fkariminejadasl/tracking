import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pylab as plt

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


def _get_dtime(
    frame_number,
    dtime_backward,
    dtime_forward,
    dtime_fb,
    dtime_limit,
    total_n_frames,
    step,
):
    dtime_limit = dtime_limit * step
    if frame_number > total_n_frames - 1 - dtime_limit:
        dtime = dtime_backward[0]
    elif frame_number < dtime_limit:
        dtime = dtime_forward[0]
    else:
        dtime = dtime_fb[0]
    dtime = dtime * step
    return dtime


def get_next_image_path(image_path: Path, tracks, dtime_limit: int = 4) -> Path:
    """
    !!!! Not a great function.
    - It assumes all the images have detections
    - It assumes training or validation set contains all images (sampled by step).
    Otherwise, _get_dtime will fail.
    """

    video_name, frame_number1 = _get_video_name_and_frame_number(image_path)

    vid_name_600 = [
        "04_07_22_F_2_rect_valid",
        "04_07_22_G_2_rect_valid",
    ]
    vid_infos = {"short": [600, 1], "long": [3117, 8]}

    dtime_forward = np.random.permutation(np.arange(1, dtime_limit + 1))
    dtime_backward = np.random.permutation(np.arange(-dtime_limit, 0))
    dtime_fb = np.random.permutation(np.hstack((dtime_forward, dtime_backward)))
    if video_name in vid_name_600:
        vid_info = vid_infos["short"]
    else:
        vid_info = vid_infos["long"]
    dtime = _get_dtime(
        frame_number1, dtime_backward, dtime_forward, dtime_fb, dtime_limit, *vid_info
    )
    print(dtime)

    frame_number2 = frame_number1 + dtime
    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    bboxes2 = tracks[tracks[:, 1] == frame_number2]
    med_disp = calculate_median_disp(tracks, frame_number1, frame_number2)
    if med_disp > accepted_disp:
        dtime = vid_info[1]

    next_image = (
        image_path.parent / f"{video_name}_frame_{frame_number1 + dtime:06d}.jpg"
    )
    print(dtime, med_disp)
    assert next_image.exists()
    return next_image, int(dtime / vid_info[1])


def get_crop_image(image_path, x_tl, y_tl, x_br, y_br):
    frame = cv2.imread(image_path.as_posix())
    c_frame = frame[y_tl:y_br, x_tl:x_br]
    return c_frame


def change_origin_bboxes(bboxes, x_tl, y_tl):
    c_bboxes = bboxes.copy()
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


def calculate_median_disp(tracks, frame_number1, frame_number2):
    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    bboxes2 = tracks[tracks[:, 1] == frame_number2]
    track_ids = set(bboxes1[:, 0]).intersection(bboxes2[:, 0])
    disps = []
    for track_id in track_ids:
        cen1 = bboxes1[bboxes1[:, 0] == track_id, 7:9][0]
        cen2 = bboxes2[bboxes2[:, 0] == track_id, 7:9][0]
        disp = np.linalg.norm(cen1 - cen2)
        disps.append(disp)
    disps = np.array(disps).astype(np.int64)
    print(sorted(dict(list(zip(track_ids, disps))).items(), key=lambda x: x[1]))
    med_disp = np.median(disps)
    return med_disp


crop_height, crop_width = 256, 512
number_bboxes = 5
np.random.seed(342)
# in attach median displacement is about 21. This is about 2 frames.
accepted_disp = 30
dtime_limit = 4


# 1. random select next image based on random d_time
image_path1 = Path(
    "/home/fatemeh/Downloads/data8_v1/train/images/183_cam_1_frame_002440.jpg"  # 183_cam_1_frame_002440, 406_cam_2_frame_001672.jpg"
)
video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)

tracks = da.load_tracks_from_mot_format(
    Path(f"/home/fatemeh/Downloads/vids/mot/{video_name}.zip")
)

image_path2, dtime = get_next_image_path(image_path1, tracks)
# image_path2 = Path(
#     "/home/fatemeh/Downloads/data8_v1/train/images/183_cam_1_frame_002456.jpg"#183_cam_1_frame_002472, 002408.jpg"  # seems this one has an issue
# )
print(image_path1)
print(image_path2)


_, frame_number2 = _get_video_name_and_frame_number(image_path2)
bboxes1 = tracks[tracks[:, 1] == frame_number1]
bboxes2 = tracks[tracks[:, 1] == frame_number2]


# 2. bbox: check(accept/reject) + select
# frame1
track_ids1 = list(np.random.permutation(bboxes1[:, 0]))
for track_id1 in track_ids1[0:1]:

    # track_id1 = track_ids1.pop()
    bbox1 = bboxes1[bboxes1[:, 0] == track_id1][0]
    center_x, center_y = bbox1[7], bbox1[8]
    print(track_id1, center_x, center_y)
    x_tl, y_tl, x_br, y_br = da.tl_br_from_cen_wh(
        center_x, center_y, crop_width, crop_height
    )

    c_frame1 = get_crop_image(image_path1, x_tl, y_tl, x_br, y_br)
    c_bbox1 = change_origin_bboxes(bbox1.reshape((1, -1)), x_tl, y_tl)

    # frame2
    c_frame2 = get_crop_image(image_path2, x_tl, y_tl, x_br, y_br)
    c_bboxes2 = get_crop_bboxes(image_path2, x_tl, y_tl, x_br, y_br, tracks)
    track_ids2 = c_bboxes2[:, 0]
    print(c_bboxes2)
    print(c_frame1.shape, c_frame2.shape)

    # adjust number of bboxes: for smaller one are zero padded, for larger ones knn used
    if (
        (track_id1 not in c_bboxes2[:, 0])
        | (c_frame1.shape != (crop_height, crop_width, 3))
        | (c_frame2.shape != (crop_height, crop_width, 3))
    ):
        print("======>", track_id1)
        continue
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

    save_dir = Path("/home/fatemeh/Downloads/test_data/overview")
    save_dir.mkdir(parents=True, exist_ok=True)
    name_stem = f"{video_name}_{frame_number1}_{frame_number2}_{dtime}_{x_tl}_{y_tl}"

    c_bboxes2_shift = c_bboxes2.copy()
    c_bboxes2_shift[:, [4, 6, 8]] = c_bboxes2[:, [4, 6, 8]] + 256
    c_bboxes12 = np.concatenate((c_bbox1, c_bboxes2_shift), axis=0)
    c_frame12 = np.concatenate((c_frame1, c_frame2), axis=0)
    visualize.plot_detections_in_image(
        da.make_dets_from_array(c_bboxes12), c_frame12, "r", "track_id"
    )
    fig = plt.gcf()
    fig.set_figwidth(4.8)
    fig.savefig(save_dir / f"{name_stem}.jpg")

    save_dir = Path("/home/fatemeh/Downloads/test_data/crops")
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite((save_dir / f"{name_stem}.jpg").as_posix(), c_frame12)
    np.savetxt(
        save_dir / f"{name_stem}.txt",
        np.concatenate((c_bbox1, c_bboxes2), axis=0),
        header="track_id,frame_number,det_id,xtl,ytl,xbr,ybr,cenx,ceny,w,h",
        delimiter=",",
        fmt="%d",
    )
