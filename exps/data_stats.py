import sys
from pathlib import Path

import pandas as pd

path = (Path(__file__).parents[1]).as_posix()
sys.path.insert(0, path)

from tracking import data_association as da
from tracking import postprocess as pp
from tracking import tracklet_operations as tk

"""
yolo_main_path = Path("/home/fatemeh/Downloads/vids/yolo")
nums_dets = dict()
for zip_file in yolo_main_path.glob("*.zip"):
    shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
    yolo_path = zip_file.parent / zip_file.stem / "obj_train_data"
    count = 0
    for yolo_file in yolo_path.glob("*.txt"):
        count += np.loadtxt(yolo_file).shape[0]
    nums_dets[zip_file.stem] = count
    shutil.rmtree(yolo_path.parent)

mot_main_path = Path("/home/fatemeh/Downloads/vids/mot")
nums_tracks = dict()
for zip_file in mot_main_path.glob("*.zip"):
    tracks = da.load_tracks_from_mot_format(zip_file)
    nums_tracks[zip_file.stem] = len(np.unique(tracks[:, 0]))


video_main_path = Path("/home/fatemeh/Downloads/vids/all")
vids_props = dict()
for video_file in video_main_path.glob("*"):
    vc = cv2.VideoCapture(video_file.as_posix())
    vc_params = visualize.get_video_parameters(vc)
    vids_props[video_file.stem] = list(map(int, map(round, vc_params)))
"""

"""
04_07_22_F_2_rect_valid
{0: 600, 1: 600, 2: 600, 3: 600, 4: 600, 5: 600, 6: 600, 7: 600, 8: 600, 9: 600, 10: 600, 11: 600, 12: 600, 13: 600, 14: 600, 15: 600, 16: 600, 17: 600, 18: 600, 19: 600, 20: 600, 21: 254, 22: 600, 23: 600, 24: 600, 25: 600, 26: 600, 27: 310, 28: 600, 29: 600, 30: 600, 31: 600, 32: 600, 33: 600, 34: 600, 35: 600, 36: 600, 37: 600, 38: 600, 39: 600, 40: 600, 41: 600, 42: 600, 43: 600, 44: 599, 45: 50}
04_07_22_G_2_rect_valid
{0: 600, 1: 600, 2: 600, 3: 600, 4: 600, 5: 164, 6: 600, 7: 600, 8: 600, 9: 600, 10: 600, 11: 600, 12: 600, 13: 600, 14: 213, 15: 600, 16: 600, 17: 600, 18: 600, 19: 600, 20: 600, 21: 600, 22: 600, 23: 600, 24: 600, 25: 586, 26: 600, 27: 600, 28: 600, 29: 600, 30: 600, 31: 570, 32: 600, 33: 600, 34: 600, 35: 600, 36: 600, 37: 536, 38: 443, 39: 321, 40: 270, 41: 164, 42: 134}
129_cam_1
{0: 3117, 1: 3117, 2: 316, 3: 3117, 4: 3117, 5: 3117, 6: 3117, 7: 3117, 8: 3117, 9: 1075, 10: 1473, 11: 1101, 12: 848, 13: 251, 14: 201}
129_cam_2
{0: 3118, 1: 3118, 2: 3118, 3: 334, 4: 3118, 5: 3118, 6: 3118, 7: 3118, 8: 3118, 9: 1064, 10: 1509, 11: 1202, 12: 844, 13: 551, 14: 201}
161_cam_1
{0: 3117, 1: 2791, 2: 1598, 3: 3117, 4: 2719, 5: 2507, 6: 2563, 7: 550, 8: 3117, 9: 1833, 10: 3117, 11: 3117, 12: 2077, 13: 322, 14: 1080, 15: 668, 16: 261, 17: 423, 18: 319, 19: 272, 20: 125}
161_cam_2
{0: 1936, 1: 1650, 2: 1666, 3: 3117, 4: 810, 5: 3117, 6: 2496, 7: 506, 8: 3117, 9: 3117, 10: 3117, 11: 1709, 12: 1733, 13: 800, 14: 1548, 15: 2090, 16: 937, 17: 1285, 18: 837, 19: 961, 20: 684, 21: 665, 22: 451}
183_cam_1
{0: 270, 1: 2100, 2: 1500, 3: 3117, 4: 3117, 5: 3117, 6: 3117, 7: 3117, 8: 3117, 9: 3117, 10: 3117, 11: 3117, 12: 3117, 13: 3117, 14: 3117, 15: 3117, 16: 3117, 17: 2070, 18: 3117, 19: 3117, 20: 3117, 21: 3117, 22: 3117, 23: 2565, 24: 2505, 25: 1974, 26: 3117, 27: 3117, 28: 1607, 29: 2666, 30: 3117, 31: 3117, 32: 1320, 33: 1613, 34: 1989, 35: 1947, 36: 1286, 37: 1228, 38: 944, 39: 804, 40: 709, 41: 702, 42: 688, 43: 686, 44: 684, 45: 682, 46: 681, 47: 669}
183_cam_2
{0: 2217, 1: 3117, 2: 3117, 3: 3117, 4: 2047, 5: 3117, 6: 3117, 7: 2013, 8: 3117, 9: 3117, 10: 3117, 11: 2520, 12: 2567, 13: 3117, 14: 3117, 15: 3117, 16: 1917, 17: 3117, 18: 3117, 19: 3117, 20: 3117, 21: 1533, 22: 3117, 23: 3117, 24: 1053, 25: 3117, 26: 266, 27: 480, 28: 270, 29: 1854, 30: 1794, 31: 323, 32: 1104, 33: 767, 34: 714, 35: 696, 36: 134, 37: 683, 38: 677, 39: 675, 40: 674, 41: 671, 42: 654, 43: 516, 44: 467, 45: 437, 46: 331}
231_cam_1
{0: 3117, 1: 3117, 2: 3117, 3: 3117, 4: 3117, 5: 3117, 6: 3117, 7: 3117, 8: 3117, 9: 1920, 10: 1139, 11: 3117, 12: 3117, 13: 3117, 14: 3117, 15: 2946, 16: 1019, 17: 3057, 18: 3055, 19: 296, 20: 215, 21: 2782, 22: 2722, 23: 2542, 24: 414, 25: 2422, 26: 836, 27: 1458, 28: 670, 29: 425, 30: 176, 31: 56}
231_cam_2
{0: 3117, 1: 3117, 2: 3117, 3: 3117, 4: 3117, 5: 3117, 6: 3117, 7: 3117, 8: 1852, 9: 776, 10: 3117, 11: 3117, 12: 3117, 13: 3117, 14: 3117, 15: 3117, 16: 3117, 17: 1453, 18: 3117, 19: 3117, 20: 3117, 21: 210, 22: 3117, 23: 2924, 24: 3117, 25: 3027, 26: 2769, 27: 2679, 28: 2581, 29: 445, 30: 210, 31: 522, 32: 539, 33: 404, 34: 61}
261_cam_1
{0: 850, 1: 1612, 2: 3117, 3: 3117, 4: 2543, 5: 1240, 6: 120, 7: 1030, 8: 1618, 9: 120, 10: 3117, 11: 2658, 12: 180, 13: 377, 14: 650, 15: 2950, 16: 3117, 17: 1658, 18: 230, 19: 1128, 20: 1685, 21: 929, 22: 746, 23: 171, 24: 685, 25: 680, 26: 644, 27: 671, 28: 670, 29: 661, 30: 378, 31: 655, 32: 654, 33: 163, 34: 651, 35: 647, 36: 641, 37: 638, 38: 629, 39: 624, 40: 620, 41: 490, 42: 90, 43: 331, 44: 231, 45: 211, 46: 91}
349_cam_1
{0: 1371, 1: 3118, 2: 3118, 3: 3118, 4: 3118, 5: 3118, 6: 3118, 7: 2918, 8: 2619, 9: 1579}
349_cam_2
{0: 3117, 1: 3117, 2: 3117, 3: 3117, 4: 3117, 5: 3117, 6: 3117, 7: 3117, 8: 2517}
406_cam_1
{0: 3118, 1: 3118, 2: 3118, 3: 3118, 4: 3118, 5: 3118, 6: 3118, 7: 524, 8: 3118, 9: 1543}
406_cam_2
{0: 3117, 1: 3117, 2: 3117, 3: 3117, 4: 3117, 5: 701, 6: 3117, 7: 3117, 8: 3117, 9: 917}
"""


#
vids_props = {
    "04_07_22_F_2_rect_valid": [1220, 2098, 600, 30],
    "04_07_22_G_2_rect_valid": [1220, 2098, 600, 30],
    "129_cam_1": [1080, 1920, 3117, 240],
    "129_cam_2": [1080, 1920, 3118, 240],
    "161_cam_1": [1080, 1920, 3117, 240],
    "161_cam_2": [1080, 1920, 3117, 240],
    "183_cam_1": [1080, 1920, 3117, 240],
    "183_cam_2": [1080, 1920, 3117, 240],
    "231_cam_1": [1080, 1920, 3117, 240],
    "231_cam_2": [1080, 1920, 3117, 240],
    "261_cam_1": [1080, 1920, 3117, 240],
    "349_cam_1": [1080, 1920, 3118, 240],
    "349_cam_2": [1080, 1920, 3117, 240],
    "406_cam_1": [1080, 1920, 3118, 240],
    "406_cam_2": [1080, 1920, 3117, 240],
}

nums_dets = {
    "04_07_22_F_2_rect_valid": 26413,
    "04_07_22_G_2_rect_valid": 22633,
    "129_cam_1": 30201,
    "129_cam_2": 30649,
    "161_cam_1": 35693,
    "161_cam_2": 38349,
    "183_cam_1": 105580,
    "183_cam_2": 86160,
    "231_cam_1": 68671,
    "231_cam_2": 82792,
    "261_cam_1": 46718,
    "349_cam_1": 27195,
    "349_cam_2": 27453,
    "406_cam_1": 27011,
    "406_cam_2": 26554,
}

nums_tracks = {
    "04_07_22_F_2_rect_valid": 46,
    "04_07_22_G_2_rect_valid": 43,
    "129_cam_1": 15,
    "129_cam_2": 15,
    "161_cam_1": 21,
    "161_cam_2": 23,
    "183_cam_1": 48,
    "183_cam_2": 47,
    "231_cam_1": 32,
    "231_cam_2": 35,
    "261_cam_1": 47,
    "349_cam_1": 10,
    "349_cam_2": 9,
    "406_cam_1": 10,
    "406_cam_2": 10,
}

data = vids_props.copy()
for key, val in nums_dets.items():
    data[key].append(val)
for key, val in nums_tracks.items():
    data[key].append(val)


data_transpos = {
    key: [] for key in ["vid", "imsize", "#frames", "fps", "#dets", "#tracks"]
}
for key, val in data.items():
    data_transpos["vid"].append(key)
    data_transpos["imsize"].append(f"{val[0]}x{val[1]}")
    data_transpos["#frames"].append(val[2])
    data_transpos["fps"].append(val[3])
    data_transpos["#dets"].append(val[4])
    data_transpos["#tracks"].append(val[5])

df = pd.DataFrame(data=data_transpos)

# generate html tables for readme.md
# with open("/home/fatemeh/Downloads/table.txt", "w") as rfile:
#     html = df.to_html()
#     rfile.write(html)

# tracks lengths
# ==============
mot_main_path = Path("/home/fatemeh/Downloads/vids/mot")
tracks_lengths = {}
for zip_file in mot_main_path.glob("*.zip"):
    tracks = da.load_tracks_from_mot_format(zip_file)
    tracks_lengths[zip_file.stem] = pp.get_tracks_lengths(tracks)

tracks_lengths_t = {key: [] for key in ["vid", "track_id:track_length"]}
for key, val in tracks_lengths.items():
    tracks_lengths_t["vid"].append(key)
    ids_lengths = ""
    for track_id, track_length in val.items():
        ids_lengths += f"{track_id}:{track_length},"
    ids_lengths = ids_lengths.strip(",")
    tracks_lengths_t["track_id:track_length"].append(f"{ids_lengths}")

df_tracks_lengths = pd.DataFrame(data=tracks_lengths_t)

# generate html tables for readme.md
# with open("/home/fatemeh/Downloads/table.txt", "w") as rfile:
#     html = df_tracks_lengths.to_html()
#     rfile.write(html)
