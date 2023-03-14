import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

import tracking.data_association as da
import tracking.visualize as visualize


def _get_video_name_and_frame_number(image_path: Path) -> tuple[str, int]:
    split_name = image_path.stem.split("_frame_")
    video_name = split_name[0]
    frame_number = int(split_name[1])
    return video_name, frame_number


def _get_dframe(
    frame_number,
    dtime_backward,
    dtime_forward,
    dtime_fb,
    dtime_limit,
    total_n_frames,
    step,
):
    dframe_limit = dtime_limit * step
    if frame_number > total_n_frames - 1 - dframe_limit:
        dtime = dtime_backward[0]
    elif frame_number < dframe_limit:
        dtime = dtime_forward[0]
    else:
        dtime = dtime_fb[0]
    dframe = dtime * step
    return dframe


def get_next_image_paths(
    image_path1: Path, tracks, dtime_limit: int = 4
) -> None | list:
    vid_name_600 = [
        "04_07_22_F_2_rect_valid",
        "04_07_22_G_2_rect_valid",
    ]
    vid_infos = {"short": [600, 1], "long": [248, 8]}  # "long": [3117, 8]}

    video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)
    if video_name in vid_name_600:
        vid_info = vid_infos["short"]
    else:
        vid_info = vid_infos["long"]

    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    if bboxes1.size == 0:
        return
    image_dir = image_path1.parent

    image_paths2_dtimes = []
    dtimes = np.hstack((np.arange(-dtime_limit, 0), np.arange(1, dtime_limit + 1)))
    for dtime in dtimes:
        frame_number2 = frame_number1 + dtime * vid_info[1]
        if (frame_number2 < 0) or (frame_number2 >= vid_info[0]):
            continue
        bboxes2 = tracks[tracks[:, 1] == frame_number2]
        if bboxes2.size == 0:
            continue
        med_disp = calculate_median_disp(tracks, frame_number1, frame_number2)
        if med_disp < accepted_disp:
            image_path2 = image_dir / f"{video_name}_frame_{frame_number2:06d}.jpg"
            if image_path2.exists():
                image_paths2_dtimes.append([image_path2, dtime])
                print(dtime)

    return image_paths2_dtimes


def get_next_image_path(image_path: Path, tracks, dtime_limit: int = 4) -> Path:
    """
    !!!! Not a great function.
    - It assumes all the images have detections
    - It assumes training or validation set contains all images (sampled by step).
    Otherwise, _get_dframe will fail.
    """

    video_name, frame_number1 = _get_video_name_and_frame_number(image_path)

    vid_name_600 = [
        "04_07_22_F_2_rect_valid",
        "04_07_22_G_2_rect_valid",
    ]
    vid_infos = {"short": [600, 1], "long": [248, 8]}  # "long": [3117, 8]}

    dtime_forward = np.random.permutation(np.arange(1, dtime_limit + 1))
    dtime_backward = np.random.permutation(np.arange(-dtime_limit, 0))
    dtime_fb = np.random.permutation(np.hstack((dtime_forward, dtime_backward)))
    if video_name in vid_name_600:
        vid_info = vid_infos["short"]
    else:
        vid_info = vid_infos["long"]
    dframe = _get_dframe(
        frame_number1, dtime_backward, dtime_forward, dtime_fb, dtime_limit, *vid_info
    )
    print(dframe)

    frame_number2 = frame_number1 + dframe
    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    bboxes2 = tracks[tracks[:, 1] == frame_number2]
    med_disp = calculate_median_disp(tracks, frame_number1, frame_number2)
    if med_disp > accepted_disp:
        dframe = vid_info[1]

    next_image = (
        image_path.parent / f"{video_name}_frame_{frame_number1 + dframe:06d}.jpg"
    )
    print(dframe, med_disp)
    assert next_image.exists()
    dtime = int(dframe / vid_info[1])
    return next_image, dtime


def get_crop_image(image_path, x_tl, y_tl, x_br, y_br):
    image = cv2.imread(image_path.as_posix())
    im_height, im_width, _ = image.shape
    cy_tl = np.clip(y_tl, 0, im_height)
    cy_br = np.clip(y_br, 0, im_height)
    cx_tl = np.clip(x_tl, 0, im_width)
    cx_br = np.clip(x_br, 0, im_width)
    c_image = image[cy_tl:cy_br, cx_tl:cx_br]
    padx_tl, pady_tl, padx_br, pady_br = map(
        abs, [cx_tl - x_tl, cy_tl - y_tl, cx_br - x_br, cy_br - y_br]
    )
    c_image = np.pad(c_image, ((pady_tl, pady_br), (padx_tl, padx_br), (0, 0)))
    return c_image


def change_origin_bboxes(bboxes, x_tl, y_tl):
    c_bboxes = bboxes.copy()
    c_bboxes[:, [3, 5, 7]] -= x_tl
    c_bboxes[:, [4, 6, 8]] -= y_tl
    return c_bboxes


def get_crop_bboxes(bboxes, x_tl, y_tl, x_br, y_br):
    c_bboxes = bboxes[
        (
            (bboxes[:, 7] >= x_tl)
            & (bboxes[:, 7] < x_br)
            & (bboxes[:, 8] >= y_tl)
            & (bboxes[:, 8] < y_br)
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
    # print(sorted(dict(list(zip(track_ids, disps))).items(), key=lambda x: x[1]))
    med_disp = np.median(disps)
    return med_disp


def save_result(save_dir, name_stem, c_bboxes2, c_bbox1, c_frame1, c_frame2):
    overview_dir = Path(save_dir / "overview")
    overview_dir.mkdir(parents=True, exist_ok=True)
    c_bboxes2_shift = c_bboxes2.copy()
    c_bboxes2_shift[:, [4, 6, 8]] = c_bboxes2[:, [4, 6, 8]] + crop_height
    c_bboxes12 = np.concatenate((c_bbox1, c_bboxes2_shift), axis=0)
    c_frame12 = np.concatenate((c_frame1, c_frame2), axis=0)
    visualize.plot_detections_in_image(
        da.make_dets_from_array(c_bboxes12), c_frame12, "r", "track_id"
    )
    fig = plt.gcf()
    fig.set_figwidth(4.8)
    fig.savefig(overview_dir / f"{name_stem}.jpg")
    plt.close()

    crops_dir = Path(save_dir / "crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite((crops_dir / f"{name_stem}.jpg").as_posix(), c_frame12)
    np.savetxt(
        crops_dir / f"{name_stem}.txt",
        np.concatenate((c_bbox1, c_bboxes2), axis=0),
        header="track_id,frame_number,det_id,xtl,ytl,xbr,ybr,cenx,ceny,w,h",
        delimiter=",",
        fmt="%d",
    )


def adjust_number_boxes(c_bboxes2, track_id1):
    if c_bboxes2.shape[0] < number_bboxes:
        c_bboxes2 = zero_padding_bboxes(c_bboxes2, number_bboxes)
    if c_bboxes2.shape[0] > number_bboxes:
        kdt = KDTree(c_bboxes2[:, 7:9])
        ind = np.where(c_bboxes2[:, 0] == track_id1)[0]
        _, inds = kdt.query(c_bboxes2[ind, 7:9], k=number_bboxes)
        c_bboxes2 = c_bboxes2[inds[0]]
    return c_bboxes2


def make_label(c_bboxes2, track_id1):
    assert c_bboxes2.shape[0] == number_bboxes
    inds = np.random.permutation(number_bboxes)
    c_bboxes2 = c_bboxes2[inds]
    c_bboxes2[:, 2] = np.arange(number_bboxes)
    lable = c_bboxes2[c_bboxes2[:, 0] == track_id1, 2]
    print("label: ", lable)


def generate_data_per_image(save_dir, image_path1, image_path2, dtime, tracks):
    video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)
    video_name, frame_number2 = _get_video_name_and_frame_number(image_path2)
    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    bboxes2 = tracks[tracks[:, 1] == frame_number2]

    # TODO: data augmentation
    track_ids1 = list(np.random.permutation(bboxes1[:, 0]))
    for track_id1 in track_ids1:
        # frame1
        bbox1 = bboxes1[bboxes1[:, 0] == track_id1][0]
        center_x, center_y = bbox1[7], bbox1[8]
        # Jitter: The crop is not only in the image center
        center_x += int(np.random.normal(jitter_loc, jitter_scale, 1))
        center_y += int(np.random.normal(jitter_loc, jitter_scale, 1))
        print("track_id1, center_x, center_y: ", track_id1, center_x, center_y)
        xy_tl_br = da.tl_br_from_cen_wh(center_x, center_y, crop_width, crop_height)

        # the cutting limits can be outside image borders. They are padded.
        c_frame1 = get_crop_image(image_path1, *xy_tl_br)
        c_bbox1 = change_origin_bboxes(bbox1.reshape((1, -1)), *xy_tl_br[:2])

        # frame2
        c_frame2 = get_crop_image(image_path2, *xy_tl_br)
        c_bboxes2 = get_crop_bboxes(bboxes2, *xy_tl_br)

        if track_id1 not in c_bboxes2[:, 0]:
            print("track_id1 not in c_bboxs2 ======>", track_id1)
            continue
        # adjust number of bboxes: for smaller one are zero padded, for larger ones knn used
        c_bboxes2 = adjust_number_boxes(c_bboxes2, track_id1)

        make_label(c_bboxes2, track_id1)

        name_stem = f"{video_name}_{frame_number1}_{frame_number2}_{dtime}_{xy_tl_br[0]}_{xy_tl_br[1]}"
        save_result(save_dir, name_stem, c_bboxes2, c_bbox1, c_frame1, c_frame2)


crop_height, crop_width = 256, 512
number_bboxes = 5
np.random.seed(342)
# in attach median displacement is about 21. This is about 2 frames.
accepted_disp = 30
dtime_limit = 4
jitter_loc = 50
jitter_scale = 10
image_dir = Path("/home/fatemeh/Downloads/test_data/images")
save_dir = image_dir.parent

# 1. random select next image based on random d_time
# image_path1 = Path(
#     "/home/fatemeh/Downloads/data8_v1/train/images/183_cam_1_frame_002440.jpg"
# )
for image_path1 in sorted(image_dir.glob("*.jpg")):
    video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)
    tracks = da.load_tracks_from_mot_format(
        Path(f"/home/fatemeh/Downloads/vids/mot/{video_name}.zip")
    )

    # image_path2, dtime = get_next_image_path(image_path1, tracks)
    image_paths2_dtimes = get_next_image_paths(image_path1, tracks)
    if image_paths2_dtimes is None:
        continue
    print(image_path1)
    for image_path2, dtime in image_paths2_dtimes:
        print(image_path2)

    # 2. generate data per image
    generate_data_per_image(save_dir, image_path1, image_path2, dtime, tracks)
    break
