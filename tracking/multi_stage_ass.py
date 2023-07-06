from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from tracking import data_association as da

np.random.seed(1000)

# e.g. overlap v=217_cam12, f=120, t1=8, t2=10
main_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids")
w_enlarge, h_enlarge = 0, 0


def bbox_enlarge(bbox, w_enlarge, h_enlarge):
    n_bbox = deepcopy(bbox)
    n_bbox[3] -= w_enlarge
    n_bbox[5] += w_enlarge
    n_bbox[4] -= h_enlarge
    n_bbox[6] += h_enlarge
    n_bbox[9] = n_bbox[5] - n_bbox[3]
    n_bbox[10] = n_bbox[6] - n_bbox[4]
    return n_bbox


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


def get_overlaps_per_vid(vid_name: str) -> list:
    # e.g. vid_name="217_cam12"
    # return [[frame_id, track_id, track_id], ...]
    tracks = da.load_tracks_from_mot_format(main_dir / f"mots/{vid_name}.zip")
    overlaps = []
    for image_path in main_dir.glob(f"images/{vid_name}*"):
        # f"images/{vid_name}_frame_{frame_number1:06d}.jpg"
        frame_number = int(image_path.stem.split("_")[-1])
        dets = tracks[tracks[:, 1] == frame_number]
        overlap = get_overlapped_dets(dets)
        if overlap:
            overlaps += overlap
    return overlaps


def get_overlaps_vids():
    # return overlaps[vid_name] = [[frame_id, track_id, track_id], ...]
    overlaps = {}
    for vid_path in main_dir.glob("vids/*mp4"):
        vid_name = vid_path.stem
        print(vid_name)
        overlap = get_overlaps_per_vid(vid_name)
        if len(overlap) != 0:
            overlap = np.array(overlap)
            overlap = overlap[np.argsort(overlap[:, 0])]
            overlaps[vid_name] = overlap
    return overlaps


def calculate_cos_sim(
    frame_number1, frame_number2, track_id1, track_id2, vid_name, **kwargs
):
    model = kwargs.get("model")
    transform = kwargs.get("transform")
    device = kwargs.get("device")

    tracks = da.load_tracks_from_mot_format(main_dir / f"mots/{vid_name}.zip")
    im1 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number1:06d}.jpg"))
    dets1 = tracks[tracks[:, 1] == frame_number1]
    im2 = cv2.imread(str(main_dir / f"images/{vid_name}_frame_{frame_number2:06d}.jpg"))
    dets2 = tracks[tracks[:, 1] == frame_number2]

    # visualize.plot_detections_in_image(dets1[:,[0,3,4,5,6]], im1);plt.show(block=False)
    # visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], im2);plt.show(block=False)

    bb1 = deepcopy(dets1[(dets1[:, 0] == track_id1) | (dets1[:, 0] == track_id2)])
    bb2 = deepcopy(dets2[(dets2[:, 0] == track_id1) | (dets2[:, 0] == track_id2)])
    bb1 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb1])
    bb2 = np.array([bbox_enlarge(bb, w_enlarge, h_enlarge) for bb in bb2])
    w, h = max(max(bb1[:, -2]), max(bb2[:, -2])), max(max(bb1[:, -1]), max(bb2[:, -1]))

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    activation = {}
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.layer1.register_forward_hook(get_activation("layer1"))
    model.layer2.register_forward_hook(get_activation("layer2"))
    model.layer3.register_forward_hook(get_activation("layer3"))
    model.layer4.register_forward_hook(get_activation("layer4"))
    _ = model(transform(im1).unsqueeze(0).to(device))

    print("concate embeddings")
    layers = ["conv1", "layer1", "layer2", "layer3"]
    output = [bb1[0, 1], bb2[0, 1]]
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
            print(f"{bb1[i,1]}, {bb2[j,1]}, {bb1[i,0]}, {bb2[j,0]}, {csim}")
    return output


def calculate_success(out):
    out = np.array(out[2:]).reshape(-1, 3)
    out = out[np.argsort(out[:, 2])]
    success = False
    if (out[-1, 0] == out[-1, 1]) & (out[-2, 0] == out[-2, 1]):
        success = True
    return success


save_path = main_dir / "tmp.txt"

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


kwargs = {"model": model, "device": device, "transform": transform}

step = 8
end_frame = 256
frame_number1 = 200  # 216  # 0
frame_number2 = 240  # 8
track_id1 = 4  # 9
track_id2 = 11  # 11
vid_name = "384_cam12"  # "217_cam12"


def get_success_per_vid(overlaps, vid_name):
    outs = []
    for item in overlaps:
        frame_number1, track_id1, track_id2 = item
        frame_number2 = frame_number1 + step
        if frame_number2 >= end_frame:
            frame_number2 = frame_number1 - step
        out = calculate_cos_sim(
            frame_number1, frame_number2, track_id1, track_id2, vid_name, **kwargs
        )
        success = calculate_success(out)
        out += [success]
        outs.append(out)
    return outs


# overlaps = get_overlaps_per_vid(vid_name)
# outs = get_success_per_vid(overlaps, vid_name)

# np.savetxt(save_path, np.array(outs), fmt="%d", delimiter=",")


overlaps_vids = get_overlaps_vids()
outs = []
for vid_name, overlaps in overlaps_vids.items():
    print(vid_name)
    out = get_success_per_vid(overlaps, vid_name)
    outs += out
np.savetxt(save_path, np.array(outs), fmt="%d", delimiter=",")


# This part is only for out of samples
# - run csim for all overlaps and check success and failure
#   - save images again (256 is missing)
#   - check for end_image
#   - save vid name (maybe append)
#   - for out of border image cut will be empty
#   - save result in a text
#   - check failure and suceess rate

# This part for everything
# - make a vids, images, images_tracks, mots
# - do the previous steps
# - put previous data, results in downloads/fish


def test_get_overlaps_per_vid():
    overlaps = get_overlaps_per_vid("16_cam12")
    expected = [[120, 14, 21], [128, 14, 21], [112, 14, 21]]
    np.testing.assert_equal(np.array(expected), np.array(overlaps))


def test_calculate_cos_sim():
    kwargs = {"model": model, "device": device, "transform": transform}
    frame_number1 = 200
    track_id1 = 4
    track_id2 = 11
    vid_name = "384_cam12"
    desired = calculate_cos_sim(
        frame_number1, 240, track_id1, track_id2, vid_name, **kwargs
    )
    expected = [200, 240, 4, 4, 85, 4, 11, 80, 11, 4, 82, 11, 11, 85]
    np.testing.assert_array_equal(np.array(desired), np.array(expected))
    desired = calculate_cos_sim(
        frame_number1, 208, track_id1, track_id2, vid_name, **kwargs
    )
    expected = [200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 90]
    np.testing.assert_array_equal(np.array(desired), np.array(expected))


def test_calculate_success():
    out = [200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 90]
    success = calculate_success(out)
    assert success == True
    out = [200, 208, 4, 4, 92, 4, 11, 83, 11, 4, 81, 11, 11, 80]
    success = calculate_success(out)
    assert success == False


def test_get_success_per_vid():
    overlaps = get_overlaps_per_vid(vid_name)
    outs = get_success_per_vid(overlaps, vid_name)
    expected = np.loadtxt(main_dir / "test.txt", delimiter=",").astype(np.int64)
    np.testing.assert_array_equal(expected, np.array(outs, dtype=np.int64))


test_calculate_success()
test_calculate_cos_sim()
test_get_overlaps_per_vid()
test_get_success_per_vid()
print("passed")
