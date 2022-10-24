from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

data_folder = Path("/home/fatemeh/data/dataset1")
result_folder = Path("/home/fatemeh/results/dataset1")
track_folder = Path("/home/fatemeh/data/dataset1/cam1_labels/cam1_labels")
vc = cv2.VideoCapture(
    (data_folder / "12_07_22_1_C_GH040468_1_cam1_rect.mp4").as_posix()
)
vc.isOpened()

# visualize detection as video
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
total_no_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vc.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    (result_folder / "output.avi").as_posix(), fourcc, fps, (width, height)
)

for frame_number in range(1, total_no_frames + 1):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    rval, frame = vc.read()

    track_path = track_folder / f"12_07_22_1_C_GH040468_1_cam1_rect_{frame_number}.txt"
    track = np.loadtxt(track_path)

    Xs = np.int64(np.round(track[:, 1] * width))
    Ys = np.int64(np.round(track[:, 2] * height))
    Ws = np.int64(np.round(track[:, 3] * width))
    Hs = np.int64(np.round(track[:, 4] * height))
    # # show as bounding boxes
    # for x, y, w, h in zip(Xs, Ys, Ws, Hs):
    #     w2 = int(w/2)
    #     h2  = int(h/2)
    #     color = tuple(int(i) for i in np.random.randint(0, 255, (3,)))
    #     cv2.rectangle(frame, (x-w2,y-h2), (x+w2,y+h2), color=color, thickness=2)

    # show as thick points
    for i in range(6):
        for j in range(6):
            frame[Ys + i, Xs + j, :] = np.array([0, 0, 255])

    out.write(frame)
out.release()


# visualize detections in frames
for frame_number in range(1, total_no_frames + 1, 90):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    rval, frame = vc.read()

    track_path = track_folder / f"12_07_22_1_C_GH040468_1_cam1_rect_{frame_number}.txt"
    track = np.loadtxt(track_path)

    plt.figure()
    plt.imshow(frame[..., ::-1])
    plt.plot(track[:, 1] * frame.shape[1], track[:, 2] * frame.shape[0], "r*")
    plt.show(block=False)

vc.release()


# visualize tracks as video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    (result_folder / "track_cam1.avi").as_posix(),
    fourcc,
    fps,
    (700, 500),  # (width, height)
)
for frame_number in range(1, total_no_frames + 1):
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    _, frame = vc.read()

    for _, track in tracks.items():
        if frame_number in track.frameids:
            color = tuple(int(round(c * 255)) for c in track.color)
            idx = np.where(np.array(track.frameids) == frame_number)[0][0]
            coord = track.coords[idx]
            w2 = int(coord.w / 2)
            h2 = int(coord.h / 2)
            cv2.rectangle(
                frame,
                (coord.x - w2, coord.y - h2),
                (coord.x + w2, coord.y + h2),
                color=color,
                thickness=2,
            )
            # show as thick points
            # for i in range(6):
            #     for j in range(6):
            #         frame[coord.y + i, coord.x + j, :] = color
    out.write(frame[700:1200, 1400:2100, :])  # cam1: 1400:2100; cam2: 1000:1700
out.release()
