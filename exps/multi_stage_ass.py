from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from tracking import data_association as da
from tracking import multi_stage_tracking as ms

np.random.seed(1000)

w_enlarge, h_enlarge = 0, 0
kwargs = ms.get_model_args()


def get_overlapped_dets(dets):
    # input track_id, frame_id, det_id, x_tl, y_tl, x_br, y_br, x_cen, y_cen, w, h
    # return frame_id, track_id, track_id
    bboxes = deepcopy(dets)
    overlaps = []
    for bbox1 in bboxes:
        bboxes = np.delete(bboxes, 0, axis=0)
        for bbox2 in bboxes:
            iou = da.get_iou(bbox1[3:7], bbox2[3:7])
            if iou > 0:
                overlaps.append([bbox1[1], bbox1[0], bbox2[0]])
    return overlaps


def get_overlaps_per_vid(main_path, vid_name: str) -> list:
    # e.g. vid_name="217_cam12"
    # return [[frame_id, track_id, track_id], ...]
    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
    overlaps = []
    for image_path in main_path.glob(f"images/{vid_name}*"):
        # f"images/{vid_name}_frame_{frame_number1:06d}.jpg"
        frame_number = int(image_path.stem.split("_")[-1])
        dets = tracks[tracks[:, 1] == frame_number]
        overlap = get_overlapped_dets(dets)
        if overlap:
            overlaps += overlap
    return overlaps


def get_overlaps_vids(main_path):
    # return overlaps[vid_name] = [[frame_id, track_id, track_id], ...]
    overlaps = {}
    for vid_path in main_path.glob("vids/*mp4"):
        vid_name = vid_path.stem
        print(vid_name)
        overlap = get_overlaps_per_vid(main_path, vid_name)
        if len(overlap) != 0:
            overlap = np.array(overlap)
            overlap = overlap[np.argsort(overlap[:, 0])]
            overlaps[vid_name] = overlap
    return overlaps


def calculate_cos_sim(
    frame_number1,
    frame_number2,
    track_id1,
    track_id2,
    vid_name,
    dets1,
    dets2,
    main_path,
    **kwargs,
):
    model = kwargs.get("model")
    transform = kwargs.get("transform")
    device = kwargs.get("device")
    activation = kwargs.get("activation")

    im1 = cv2.imread(
        str(main_path / f"images/{vid_name}_frame_{frame_number1:06d}.jpg")
    )
    im2 = cv2.imread(
        str(main_path / f"images/{vid_name}_frame_{frame_number2:06d}.jpg")
    )

    bb1 = deepcopy(dets1[(dets1[:, 0] == track_id1) | (dets1[:, 0] == track_id2)])
    bb2 = deepcopy(dets2[(dets2[:, 0] == track_id1) | (dets2[:, 0] == track_id2)])
    if len(bb2) != 2:
        return None
    bb1 = np.array([ms.bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb1])
    bb2 = np.array([ms.bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb2])
    w, h = max(max(bb1[:, -2]), max(bb2[:, -2])), max(max(bb1[:, -1]), max(bb2[:, -1]))

    im_height, im_width, _ = im1.shape
    ms.clip_bboxs(bb1, im_height, im_width)
    ms.clip_bboxs(bb2, im_height, im_width)

    # print("concate embeddings")
    layers = ["conv1", "layer1", "layer2", "layer3"]
    output = [int(vid_name.split("_")[0]), bb1[0, 1], bb2[0, 1]]
    for i in [0, 1]:
        for j in [0, 1]:
            imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
            imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
            imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
            imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
            _ = model(transform(imc1).unsqueeze(0).to(device))
            f1 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            _ = model(transform(imc2).unsqueeze(0).to(device))
            f2 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            csim = int(np.round(csim * 100))  # percent
            output.extend([bb1[i, 0], bb2[j, 0], csim])
            # print(f"{bb1[i,1]}, {bb2[j,1]}, {bb1[i,0]}, {bb2[j,0]}, {csim}")
    return output


def calculate_success(out):
    out = np.array(out[3:]).reshape(-1, 3)
    out = out[np.argsort(out[:, 2])]
    success = False
    if (out[-1, 0] == out[-1, 1]) & (out[-2, 0] == out[-2, 1]):
        success = True
    return success


def get_success_per_vid(overlaps, vid_name, main_path, step, end_frame):
    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
    outs = []
    for item in overlaps:
        frame_number1, track_id1, track_id2 = item
        frame_number2 = frame_number1 + step
        if frame_number2 > end_frame:
            continue
        dets1 = tracks[tracks[:, 1] == frame_number1]
        dets2 = tracks[tracks[:, 1] == frame_number2]
        out = calculate_cos_sim(
            frame_number1,
            frame_number2,
            track_id1,
            track_id2,
            vid_name,
            dets1,
            dets2,
            main_path,
            **kwargs,
        )
        if out:
            success = calculate_success(out)
            out += [success]
            outs.append(out)
    return outs


def test_get_overlaps_per_vid():
    main_path = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids")
    overlaps = get_overlaps_per_vid(main_path, "16_cam12")
    expected = [[120, 14, 21], [128, 14, 21], [112, 14, 21]]
    np.testing.assert_equal(np.array(expected), np.array(overlaps))


def test_calculate_cos_sim():
    main_path = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids")
    vid_name = "384_cam12"
    frame_number1 = 200
    frame_number2 = 240
    track_id1 = 4
    track_id2 = 11

    tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
    dets1 = tracks[tracks[:, 1] == frame_number1]
    dets2 = tracks[tracks[:, 1] == frame_number2]
    desired = calculate_cos_sim(
        frame_number1,
        frame_number2,
        track_id1,
        track_id2,
        vid_name,
        dets1,
        dets2,
        main_path,
        **kwargs,
    )
    expected = [384, 200, 240, 4, 4, 85, 4, 11, 80, 11, 4, 82, 11, 11, 85]
    np.testing.assert_array_equal(np.array(desired), np.array(expected))

    frame_number2 = 208
    dets2 = tracks[tracks[:, 1] == frame_number2]
    desired = calculate_cos_sim(
        frame_number1,
        frame_number2,
        track_id1,
        track_id2,
        vid_name,
        dets1,
        dets2,
        main_path,
        **kwargs,
    )
    expected = [384, 200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 90]
    np.testing.assert_array_equal(np.array(desired), np.array(expected))


def test_calculate_success():
    out = [384, 200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 90]
    success = calculate_success(out)
    assert success == True
    out = [384, 200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 80]
    success = calculate_success(out)
    assert success == False


def test_get_success_per_vid():
    step, end_frame = 8, 256
    main_path = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids")
    vid_name = "384_cam12"
    overlaps = get_overlaps_per_vid(main_path, vid_name)
    outs = get_success_per_vid(overlaps, vid_name, main_path, step, end_frame)
    expected = np.loadtxt(main_path / "test.txt", delimiter=",").astype(np.int64)
    np.testing.assert_array_equal(expected, np.array(outs, dtype=np.int64))


test_calculate_cos_sim()
test_get_overlaps_per_vid()
test_calculate_success()
test_get_success_per_vid()
print("passed")

"""
# Run for all videos

# main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")
# save_path = main_path / f"{main_path.name}.txt"
# step = 8
# end_frame = 3112
main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/30hz")
save_path = main_path / f"{main_path.name}.txt"
step = 1
end_frame = 599
overlaps_vids = get_overlaps_vids(main_path)
with open(save_path, "a") as afile:
    for vid_name, overlaps in tqdm(overlaps_vids.items()):
        print(vid_name, len(overlaps))
        out = get_success_per_vid(overlaps, vid_name, main_path, step, end_frame)
        np.savetxt(afile, np.array(out), fmt="%d", delimiter=",")

# len(a), len(b), len(c) -> len(c) / len(a)
# out_of_sample (29, 33, 392 -> no occlusion)
# (832, 88, 744) -> 89 %
# 240hz vids in_sample
# (2460, 160, 2300) -> 93%
# 30hz vids in_sample
# (1589, 4, 1585) -> 99.7

# a = np.loadtxt(main_path/f"{main_path.name}.txt", delimiter=",").astype(np.int64)
# c = deepcopy(a[a[:,-1]==1])
# b = deepcopy(a[a[:,-1]==0])
# b = np.array(sorted(b, key=lambda x: (x[0],x[1],x[2],x[6],x[7])))
# np.savetxt(main_path/f"failed_{main_path.name}.txt", b, delimiter=',',fmt="%d")

# This part is only for out of samples
# - calculate total number of detections, total number of occlusions

# This part for everything
# - put previous data, results in downloads/fish

# s: stage, p: problem
# multi-stage association: s1:Hungerian -> s2:cosine similarity -> s3:low quality tracks
# problems to be dealt in multi-stage association
# 1. multiple detection
# 2. no detection
# 3. short fully occluded
# 4. fast motion (attack time)
# 5. partial occlusion
# - p1 will be in s2. In s2, I need to condition on it. If the cossim is too closeby, I reject the detection.
# - p2 and p3 are the same. I can discard them in s1, based on predicted location (loc and iou)
# - p4 the same as p2/p3 should be solved in s1, based on prediction.
# First implementation on the ground truth data. Only s1 and s2 will be implemented and p5 will be solved.


# [visualize.save_video_as_images(main_path/"images", vid_path, step=8) for vid_path in main_path.glob("vids/*mp4")]
# for vid_path in main_path.glob("vids/*mp4"):
#     vid_name = vid_path.stem
#     tracks = da.load_tracks_from_mot_format(main_path / f"mots/{vid_name}.zip")
#     visualize.save_video_with_tracks_as_images(
#         main_path / "images_tracks",
#         vid_path,
#         tracks,
#         start_frame=0,
#         end_frame=256,
#         step=8,
#         format="06d",
#     )

# from pathlib import Path
# import shutil
# main_path = Path("/home/fatemeh/Downloads/fish/vids/all")
# vids = sorted(set([v.stem for v in main_path.glob("*")]))
# vtoi = {v:i for i, v in enumerate(vids)}

# main_path = Path("/home/fatemeh/Downloads/fish/in_sample_vids/240hz")
# for v in main_path.glob("vids/*"):
#     mot_path = v.parent.parent / "mots"
#     shutil.move(v, v.parent / f"{vtoi[v.stem]}.mp4")
#     shutil.move(mot_path / f"{v.stem}.zip", mot_path / f"{vtoi[v.stem]}.zip")
"""

"""
# Tests for appearance cosine similarity
def bbox_enlarge(bbox, w_enlarge, h_enlarge):
    n_bbox = deepcopy(bbox)
    n_bbox[3] -= w_enlarge
    n_bbox[5] += w_enlarge
    n_bbox[4] -= h_enlarge
    n_bbox[6] += h_enlarge
    n_bbox[9] = n_bbox[5] - n_bbox[3]
    n_bbox[10] = n_bbox[6] - n_bbox[4]
    return n_bbox


from copy import deepcopy
from importlib import reload
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import torch
import torchvision

from tracking import data_association as da
from tracking import visualize

main_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids")
w_enlarge, h_enlarge = 0, 0
# vid_name, frame_number1, frame_number2, track_id1, track_id2 = 384, 200, 240, 4, 11 #4, 4, 85, 4, 11, 80, 11, 4, 82, 11, 11, 85
vid_name, frame_number1, frame_number2, track_id1, track_id2 = 432, 120, 192, 0, 3

device = "cuda"
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
).to(device)
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

model.eval()
model.requires_grad_(False)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.conv1.register_forward_hook(get_activation("conv1"))
model.layer1.register_forward_hook(get_activation("layer1"))
model.layer2.register_forward_hook(get_activation("layer2"))
model.layer3.register_forward_hook(get_activation("layer3"))
model.layer4.register_forward_hook(get_activation("layer4"))


vid_name = f"{vid_name}_cam12"
tracks = da.load_tracks_from_mot_format(main_dir / f"mots/{vid_name}.zip")
im1 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
dets1 = tracks[tracks[:, 1] == frame_number1]
im2 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
dets2 = tracks[tracks[:, 1] == frame_number2]

bb1 = deepcopy(dets1[(dets1[:, 0] == track_id1) | (dets1[:, 0] == track_id2)])
bb2 = deepcopy(dets2[(dets2[:, 0] == track_id1) | (dets2[:, 0] == track_id2)])
bb1 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb1])
bb2 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb2])
w, h = max(max(bb1[:, -2]), max(bb2[:, -2])), max(max(bb1[:, -1]), max(bb2[:, -1]))

visualize.plot_detections_in_image(bb1[:, [0, 3, 4, 5, 6]], im1)
plt.show(block=False)
visualize.plot_detections_in_image(bb2[:, [0, 3, 4, 5, 6]], im2)
plt.show(block=False)

print("color values")
for i in [0, 1]:
    for j in [0, 1]:
        imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
        imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
        imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
        imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
        imc1 = imc1.flatten().astype(np.float32)
        imc2 = imc2.flatten().astype(np.float32)
        csim = imc1.dot(imc2) / (np.linalg.norm(imc1) * np.linalg.norm(imc2))
        print(i, j, csim)


print("embeddings")
layer = "layer2"  # layer2, and layer3 are best
for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
    print(layer)
    for i in [0, 1]:
        for j in [0, 1]:
            imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
            imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
            imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
            imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
            _ = model(transform(imc1).unsqueeze(0).to(device))
            f1 = activation[layer].flatten().cpu().numpy()
            # f1 = activation[layer][0, :, 0, 0].cpu().numpy();print(activation[layer].shape)
            _ = model(transform(imc2).unsqueeze(0).to(device))
            f2 = activation[layer].flatten().cpu().numpy()
            csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            print(i, j, csim)

print("concate embeddings")
layers = ["conv1", "layer1", "layer2", "layer3"]
for i in [0, 1]:
    for j in [0, 1]:
        imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
        imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
        imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
        imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
        _ = model(transform(imc1).unsqueeze(0).to(device))
        f1 = np.concatenate(
            [activation[layer].flatten().cpu().numpy() for layer in layers]
        )
        _ = model(transform(imc2).unsqueeze(0).to(device))
        f2 = np.concatenate(
            [activation[layer].flatten().cpu().numpy() for layer in layers]
        )
        csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        print(i, j, csim)
print(bb1)
print(bb2)
print("finished")
"""

"""
# TODO aggregate multiple frames: it should be implemented
vid_name = f"{vid_name}_cam12"
tracks = da.load_tracks_from_mot_format(main_dir / f"mots/{vid_name}.zip")
im1 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
dets1 = tracks[tracks[:, 1] == frame_number1]
bb1 = deepcopy(dets1[(dets1[:, 0] == track_id1) | (dets1[:, 0] == track_id2)])
bb1 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb1])

max_w, max_h = 0, 0
for frame_number2 in range(frame_number1, frame_number1 + 65, 8):
    dets2 = tracks[tracks[:, 1] == frame_number2]
    bb2 = deepcopy(dets2[(dets2[:, 0] == track_id1) | (dets2[:, 0] == track_id2)])
    bb2 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb2])
    w, h = max(max(bb1[:, -2]), max(bb2[:, -2])), max(max(bb1[:, -1]), max(bb2[:, -1]))
    if w > max_w:
        max_w = w
    if h > max_h:
        max_h = h
w, h = max_w, max_h

c_mem = 0
emb_mem = 0
agg_mem = 0
for frame_number2 in range(frame_number1, frame_number1 + 65, 8):
    im2 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
    dets2 = tracks[tracks[:, 1] == frame_number2]
    bb2 = deepcopy(dets2[(dets2[:, 0] == track_id1) | (dets2[:, 0] == track_id2)])
    bb2 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb2])

    print("color values")
    for i in [0, 1]:
        for j in [0, 1]:
            imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
            imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
            imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
            imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
            imc1 = imc1.flatten().astype(np.float32)
            imc2 = imc2.flatten().astype(np.float32)
            c_mem += imc2
            csim = imc1.dot(imc2) / (np.linalg.norm(imc1) * np.linalg.norm(imc2))
            print(i, j, csim)



    print("embeddings")
    layer = "layer2"  # layer2, and layer3 are best
    for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
        print(layer)
        for i in [0, 1]:
            for j in [0, 1]:
                imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
                imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
                imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
                imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
                _ = model(transform(imc1).unsqueeze(0).to(device))
                f1 = activation[layer].flatten().cpu().numpy()
                # f1 = activation[layer][0, :, 0, 0].cpu().numpy();print(activation[layer].shape)
                _ = model(transform(imc2).unsqueeze(0).to(device))
                f2 = activation[layer].flatten().cpu().numpy()
                csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                print(i, j, csim)

    print("concate embeddings")
    layers = ["conv1", "layer1", "layer2", "layer3"]
    for i in [0, 1]:
        for j in [0, 1]:
            imc1 = im1[bb1[i, 4] : bb1[i, 6], bb1[i, 3] : bb1[i, 5]]
            imc2 = im2[bb2[j, 4] : bb2[j, 6], bb2[j, 3] : bb2[j, 5]]
            imc1 = cv2.resize(imc1, (w, h), interpolation=cv2.INTER_AREA)
            imc2 = cv2.resize(imc2, (w, h), interpolation=cv2.INTER_AREA)
            _ = model(transform(imc1).unsqueeze(0).to(device))
            f1 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            _ = model(transform(imc2).unsqueeze(0).to(device))
            f2 = np.concatenate(
                [activation[layer].flatten().cpu().numpy() for layer in layers]
            )
            csim = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            print(i, j, csim)
    print(bb1)
    print(bb2)
print("finished")
"""
