from pathlib import Path

import numpy as np
from tqdm import tqdm

from tracking import data_association as da
from tracking import visualize as tv

# split xml to mots
xml_path = Path("/home/fatemeh/Downloads/fish/mot_data/loom_video_annotations_sara.xml")
save_path = Path("/home/fatemeh/Downloads/fish/mot_data/mots")
da.xml_to_mots(xml_path, save_path)

# split stacked videos to two videos
save_path = Path("/home/fatemeh/Downloads/fish/mot_data/vids")
vid_files = list(
    Path("/home/fatemeh/Downloads/fish/mot_data/stacked_vids").glob("*mp4")
)
for vid_file in vid_files:
    tv.split_video(vid_file, save_path)

# add: 161, 349; use longer vids: {129, 183, 231, 261_1, 406} from in_sample
# {328, 335, 336, 340, 341, 347} : no videos available
# {44, 147, 222, 235, 390}: no annotation available
# {72, 298, 311, 313, 314, 393, 397, 400, 408, 433} is empty annotations
# don't use 327 through 350
# stats based on xml (short videos):
# 29,639 frames, 3,998 tracks, 969,959 boxes
# final stats
# 46,280 frames, 3,688 tracks, 887,932 boxes (for short videos)
# 84,464 frames, 3,796 tracks, 1,469,616 boxes (short with long ones)
# short videos: 1080 x 1920 resolution, 240HZ, 260 frames
# long videos : 1080 x 1920 resolution, 240HZ, 3117 frames (129_1, 161, 183, 231, 261_1, 349, 406_2), 318 frames(129_2, 349_1, 406_1)

n_frames = 0
n_tracks = 0
n_bboxes = 0
m = Path("/home/fatemeh/Downloads/fish/mot_data/mots")
for p in tqdm(m.glob("*txt")):
    tracks = np.loadtxt(p, delimiter=",", dtype=np.int64)
    n_frames += len(np.unique(tracks[:, 0]))
    n_tracks += len(np.unique(tracks[:, 1]))
    n_bboxes += tracks.shape[0]

print(f"{n_frames:,} frames, {n_tracks:,} tracks, {n_bboxes:,} boxes")
