from copy import deepcopy
from pathlib import Path

import numpy as np

from tracking import data_association as da

np.random.seed(1000)


def bbox_enlarge(bbox, w_enlarge, h_enlarge):
    n_bbox = deepcopy(bbox)
    n_bbox[3] -= w_enlarge
    n_bbox[5] += w_enlarge
    n_bbox[4] -= h_enlarge
    n_bbox[6] += h_enlarge
    n_bbox[9] = n_bbox[5] - n_bbox[3]
    n_bbox[10] = n_bbox[6] - n_bbox[4]
    return n_bbox


# e.g. overlap v=217_cam12, f=120, t1=8, t2=10
main_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids")


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


overlaps = {}
for vid_path in main_dir.glob(f"vids/*mp4"):
    vid_name = vid_path.stem
    print(vid_name)
    overlap = get_overlaps_per_vid(vid_name)
    if len(overlap) != 0:
        overlap = np.array(overlap)
        overlap = overlap[np.argsort(overlap[:, 0])]
        overlaps[vid_name] = overlap

# This part is only for out of samples
# - make csim as function
# - run csim for all overlaps and check success and failure

# This part for everything
# - make a vids, images, images_tracks, mots
# - do the previous steps
# - put previous data, results in downloads/fish

"""
frame_number1 = 200#216  # 0
frame_number2 = 240  # 8
for frame_number2 in [208, 216, 224, 232, 240, 248, 256]:
    print(frame_number2)
    track_id1 = 4  # 9
    track_id2 = 11  # 11
    w_enlarge, h_enlarge = 0, 0
    vid_name = "384_cam12"  # "217_cam12"
    main_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids")
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


    activation = {}


    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook


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
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.layer1.register_forward_hook(get_activation("layer1"))
    model.layer2.register_forward_hook(get_activation("layer2"))
    model.layer3.register_forward_hook(get_activation("layer3"))
    model.layer4.register_forward_hook(get_activation("layer4"))
    _ = model(transform(im1).unsqueeze(0).to(device))


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
"""
