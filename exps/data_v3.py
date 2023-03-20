import sys
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import torchvision
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


def get_crop_image(image, x_tl, y_tl, x_br, y_br):
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


def save_result(save_dir, name_stem, c_image1, c_bbox1, c_image2, c_bboxes2):
    overview_dir = Path(save_dir / "overview")
    overview_dir.mkdir(parents=True, exist_ok=True)
    c_bboxes2_shift = c_bboxes2.copy()
    c_bboxes2_shift[:, [4, 6, 8]] = c_bboxes2[:, [4, 6, 8]] + crop_height
    c_bboxes12 = np.concatenate((c_bbox1, c_bboxes2_shift), axis=0)
    c_image12 = np.concatenate((c_image1, c_image2), axis=0)
    visualize.plot_detections_in_image(
        da.make_dets_from_array(c_bboxes12), c_image12, "r", "track_id"
    )
    fig = plt.gcf()
    fig.set_figwidth(4.8)
    fig.savefig(overview_dir / f"{name_stem}.jpg")
    plt.close()

    crops_dir = Path(save_dir / "crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite((crops_dir / f"{name_stem}.jpg").as_posix(), c_image12)
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


def reconstruct_bboxes(bboxs, ids):
    """bboxs is array with track_id, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl
    It returns: track_id, frame_number, id, x_tl, y_tl, x_br, y_br, x_cen, y_cen, w, h
    """
    bboxs = np.array(
        [
            (
                min(bbox[1], bbox[3], bbox[5], bbox[7]),
                min(bbox[2], bbox[4], bbox[6], bbox[8]),
                max(bbox[1], bbox[3], bbox[5], bbox[7]),
                max(bbox[2], bbox[4], bbox[6], bbox[8]),
            )
            for bbox in bboxs
        ]
    )
    cen_whs = np.array(
        [da.cen_wh_from_tl_br(*bbox) for bbox in bboxs]
    ).astype(np.int64)
    bboxs = bboxs.astype(np.int64)
    return np.concatenate((ids, bboxs, cen_whs), axis=1)


def deconstruct_bboxes(bboxs):
    """from bboxes take four corners: xy_tl, xy_tr, xy_br, xy_bl
    bbox: track_id, frame_number, id, x_tl, y_tl, x_br, y_br, x_cen, y_cen, w, h
    return: track_id, xy_tl, xy_tr, xy_br, xy_bl"""

    return np.array(
        [
            (
                bbox[0],
                bbox[3],
                bbox[4],
                bbox[3] + bbox[9],
                bbox[4],
                bbox[5],
                bbox[6],
                bbox[3],
                bbox[4] + bbox[10],
            )
            for bbox in bboxs
        ]
    ).astype(np.int64)


def transform_image_bboxes(im: np.ndarray, bboxs, a=10, tt=np.array([30, 10]), s=0.75):
    im_cen_x, im_cen_y = im.shape[1] // 2, im.shape[0] // 2
    im_cen = np.array([im_cen_x, im_cen_y])
    cos = np.cos(np.deg2rad(a))
    sin = np.sin(np.deg2rad(a))
    r = s * np.array([[cos, -sin], [sin, cos]])

    ids = bboxs[:, :3].copy()
    bboxs = deconstruct_bboxes(bboxs)
    tbboxs = np.concatenate(
        (
            bboxs[:, 0:1],
            (bboxs[:, 1:3] - im_cen) @ r + tt + im_cen,
            (bboxs[:, 3:5] - im_cen) @ r + tt + im_cen,
            (bboxs[:, 5:7] - im_cen) @ r + tt + im_cen,
            (bboxs[:, 7:9] - im_cen) @ r + tt + im_cen,
        ),
        axis=1,
    )
    tbboxs = reconstruct_bboxes(tbboxs, ids)
    im = Image.fromarray(im)  # opencv to pil
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    tim = torchvision.transforms.functional.affine(
        im, -a, tuple(tt), s, 0, interpolation=interpolation
    )
    return np.array(tim), tbboxs


def geometric_transformation(bbox1):
    center_x, center_y = bbox1[7], bbox1[8]
    # Jitter: The crop is not only in the image center
    center_x += int(np.random.normal(jitter_loc, jitter_scale, 1))
    center_y += int(np.random.normal(jitter_loc, jitter_scale, 1))
    xy_tl_br = da.tl_br_from_cen_wh(center_x, center_y, crop_width, crop_height)
    return xy_tl_br


def trans_augmentation(image1, bbox1, image2, bboxes2, save_dir, name_stem):
    xy_tl_br = geometric_transformation(bbox1)
    # the cutting limits can be outside image borders. They are padded.
    c_image1 = get_crop_image(image1, *xy_tl_br)
    c_bbox1 = change_origin_bboxes(bbox1.reshape((1, -1)), *xy_tl_br[:2])
    c_image2 = get_crop_image(image2, *xy_tl_br)
    c_bboxes2 = get_crop_bboxes(bboxes2, *xy_tl_br)

    post_augmentation(
        c_image1, c_bbox1, c_image2, c_bboxes2, xy_tl_br, save_dir, name_stem
    )


def post_augmentation(
    c_image1, c_bbox1, c_image2, c_bboxes2, xy_tl_br, save_dir, name_stem
):
    """all steps needed after augmentation to save data"""

    track_id1 = c_bbox1[0, 0]
    if track_id1 not in c_bboxes2[:, 0]:
        print("track_id1 not in c_bboxs2 ======>", track_id1)
        return

    # adjust number of bboxes: for smaller one are zero padded, for larger ones knn used
    c_bboxes2 = adjust_number_boxes(c_bboxes2, track_id1)

    make_label(c_bboxes2, track_id1)

    name_stem = f"{name_stem}_{xy_tl_br[0]}_{xy_tl_br[1]}"
    save_result(save_dir, name_stem, c_image1, c_bbox1, c_image2, c_bboxes2)


def generate_data_per_image(save_dir, image_path1, image_path2, dtime, tracks):
    video_name, frame_number1 = _get_video_name_and_frame_number(image_path1)
    video_name, frame_number2 = _get_video_name_and_frame_number(image_path2)
    image1 = cv2.imread(image_path1.as_posix())
    image2 = cv2.imread(image_path2.as_posix())
    bboxes1 = tracks[tracks[:, 1] == frame_number1]
    bboxes2 = tracks[tracks[:, 1] == frame_number2]
    name_stem = f"{video_name}_{frame_number1}_{frame_number2}_{dtime}"
    print("image_path1: ", image_path1)
    print("image_path2: ", image_path2)

    for bbox1 in bboxes1:
        # data augmentation
        for i in range(1):
            trans_augmentation(image1, bbox1, image2, bboxes2, save_dir, name_stem)
            trans_augmentation(image1, bbox1, image2, bboxes2, save_dir, name_stem)


crop_height, crop_width = 256, 512
number_bboxes = 5
np.random.seed(3421)
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
for image_path1 in tqdm(sorted(image_dir.glob("*.jpg"))):
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
    # generate_data_per_image(save_dir, image_path1, image_path2, dtime, tracks)
    break
