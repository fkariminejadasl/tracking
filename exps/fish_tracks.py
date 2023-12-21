from pathlib import Path

from tracking import data_association as da
from tracking import visualize as tv

# split xml to mots
xml_path = Path(
    "/home/fatemeh/Downloads/fish/short_videos/loom_video_annotations_sara.xml"
)
save_path = Path("/home/fatemeh/Downloads/fish/short_videos/mots")
da.xml_to_mots(xml_path, save_path)

# split stacked videos to two videos
save_path = Path("/home/fatemeh/Downloads/fish/short_videos/vids")
vid_files = list(
    Path("/home/fatemeh/Downloads/fish/short_videos/stacked_vids").glob("*mp4")
)
for vid_file in vid_files:
    tv.split_video(vid_file, save_path)

# TODO clean this data (task_id_to_name from xml_to_mots)
# TODO get stats
# add: 161; use longer vids: {129, 183, 231, 249, 406} from in_sample
# {328, 335, 336, 340, 341, 347} : no videos available
# {390, 235, 44, 147, 222}: no annotation available
# don't use 327 through 350
# stats based on xml:
# 29,639 frames, 3,998 tracks, 969,959 boxes, 260 frames (short vids),1920x1080 resolution, 240HZ
