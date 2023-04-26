from copy import deepcopy
import cv2
from PIL import Image
import torch, torchvision
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KDTree
from tracking import association_learning as al, visualize, data_association as da

def normalize_bboxs(bboxs, crop_w, crop_h):
    assert bboxs.shape[1] == 5
    bboxs = bboxs.astype(np.float32)
    return np.concatenate((bboxs[:, 0:1], bboxs[:, 1:]/np.tile(np.array([crop_w, crop_h]), 2)), axis=1)


def change_center_bboxs(bboxs, crop_x, crop_y):
    assert bboxs.shape[1] == 5
    return np.concatenate((bboxs[:, 0:1], bboxs[:, 1:]-np.tile(np.array([crop_x, crop_y]),2)), axis=1)

def zero_out_of_image_bboxs(bboxs, crop_w, crop_h):
    assert bboxs.shape[1] == 5
    inds = np.where((bboxs[:,1]>=crop_w) | (bboxs[:,1]<0)| (bboxs[:,3]>=crop_w) | (bboxs[:,3]<0) | (bboxs[:,2]>=crop_h) | (bboxs[:,2]<0) | (bboxs[:,4]>=crop_h) | (bboxs[:,4]<0))[0]
    bboxs[inds, 1:] = 0
    return bboxs

def zero_padding_images(image, crop_w, crop_h):
    pad_x = max(0, crop_w - image.shape[1])
    pad_y = max(0, crop_h - image.shape[0])
    image = np.pad(image, ((0, pad_y), (0, pad_x), (0,0)))
    return image

step = 8
frame_number = 64
time = 1
frame_number2 = frame_number + step * time
im = cv2.imread(f"/home/fatemeh/Downloads/fish/data8_v3/train/images/406_cam_1_frame_{frame_number:06d}.jpg")[:,:,::-1]
dets = da.get_detections_array(Path(f"/home/fatemeh/Downloads/fish/data8_v3/train/labels/406_cam_1_frame_{frame_number:06d}.txt"), im.shape[1], im.shape[0])
im2 = cv2.imread(f"/home/fatemeh/Downloads/fish/data8_v3/train/images/406_cam_1_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets2 = da.get_detections_array(Path(f"/home/fatemeh/Downloads/fish/data8_v3/train/labels/406_cam_1_frame_{frame_number2:06d}.txt"), im.shape[1], im.shape[0])

device = 'cuda'
model = al.AssociationNet(512, 5).to(device)
model.load_state_dict(torch.load("/home/fatemeh/Downloads/result_snellius/al/1_best.pth")["model"])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
crop_w, crop_h = 512, 256
kdt = KDTree(dets2[:, 7:9])
_, inds = kdt.query(dets2[:,7:9], k=5)

for ind in inds:
    for time in range(-4,5):
        frame_number2 = frame_number + step * time
        im2 = cv2.imread(f"/home/fatemeh/Downloads/fish/data8_v3/train/images/406_cam_1_frame_{frame_number2:06d}.jpg")[:,:,::-1]
        dets2 = da.get_detections_array(Path(f"/home/fatemeh/Downloads/fish/data8_v3/train/labels/406_cam_1_frame_{frame_number2:06d}.txt"), im.shape[1], im.shape[0])


        # ind = [7, 1, 3, 4, 2]
        bbox1 = deepcopy(dets[ind[0], 2:7][None,:])
        crop_x, crop_y = max(0, int(bbox1[0, 1]-crop_w/2)), max(0, int(bbox1[0, 2]-crop_h/2))
        bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
        bboxes2 = deepcopy(dets2[ind, 2:7])
        bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y)

        detc = bbox1.copy()
        detsc2 = bboxes2.copy()

        bboxes2 = zero_out_of_image_bboxs(bboxes2, crop_w, crop_h)
        bbox1 = normalize_bboxs(bbox1, crop_w, crop_h)
        bboxes2 = normalize_bboxs(bboxes2, crop_w, crop_h)
        # print(bboxes2)
        bbox1 = torch.tensor(bbox1[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        bboxes2 = torch.tensor(bboxes2[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        time = torch.tensor([time/4], dtype=torch.float32).to(device)

        imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc = np.ascontiguousarray(imc)
        imc2 = np.ascontiguousarray(imc2)
        imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
        imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)
        output = model(imt, bbox1, imt2, bboxes2, time)
        print(frame_number2, ind, torch.argmax(output, axis=1).item(), list(output.detach().cpu().numpy()[0]))
# print(detsc2)

# visualize.plot_detections_in_image(detc, imc[...,::-1]);plt.show(block=False)
# visualize.plot_detections_in_image(detsc2, imc2[...,::-1]);plt.show(block=False)


"""
im = cv2.imread("/home/fatemeh/Downloads/fish/data8_v3/train/images/406_cam_1_frame_000064.jpg")[:,:,::-1]
im2 = cv2.imread("/home/fatemeh/Downloads/fish/data8_v3/train/images/406_cam_1_frame_000072.jpg")[:,:,::-1]
dets = da.get_detections_array(Path("/home/fatemeh/Downloads/fish/data8_v3/train/labels/406_cam_1_frame_000064.txt"), im.shape[1], im.shape[0])
dets2 = da.get_detections_array(Path("/home/fatemeh/Downloads/fish/data8_v3/train/labels/406_cam_1_frame_000072.txt"), im.shape[1], im.shape[0])

kdt = KDTree(dets2[:, 7:9])
_, inds = kdt.query(dets2[:,7:9], k=5)

device = 'cuda'
model = al.AssociationNet(512, 5).to(device)
model.load_state_dict(torch.load("/home/fatemeh/Downloads/result_snellius/al/1_best.pth")["model"])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

crop_w, crop_h = 512, 256

crop_x, crop_y = 500, 200 # 675, 290
ind = [2,3,4,5,1]

imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
imc = np.ascontiguousarray(imc)
imc2 = np.ascontiguousarray(imc2)
imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)

bbox1 = deepcopy(dets[2:3, 3:7]-np.tile(np.array([crop_x, crop_y]),2)).astype(np.float32)
bboxes2 = deepcopy(dets2[ind, 3:7]-np.tile(np.array([crop_x, crop_y]),2)).astype(np.float32)
bbox1 /= np.tile(np.array([crop_w, crop_h]), 2)
bboxes2 /= np.tile(np.array([crop_w, crop_h]), 2)
bbox1 = torch.tensor(bbox1).unsqueeze(0).to(device)
bboxes2 = torch.tensor(bboxes2).unsqueeze(0).to(device)
# bboxes2[0,-1] = 0.0
time = torch.tensor([1/4], dtype=torch.float32).to(device)

model(imt, bbox1, imt2, bboxes2, time)

detsc = np.concatenate((dets[2:3, :3], dets[2:3, 3:9]-np.tile(np.array([crop_x, crop_y]),3), dets[2:3, 9:]), axis=1)
detsc2 = np.concatenate((dets2[ind, :3], dets2[ind, 3:9]-np.tile(np.array([crop_x, crop_y]),3), dets2[ind, 9:]), axis=1)
visualize.plot_detections_in_image(da.make_dets_from_array(detsc), imc[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(da.make_dets_from_array(detsc2), imc2[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(dets[:,2:7], im[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(dets2[:,2:7], im2[...,::-1]);plt.show(block=False)

dataset = al.AssDataset(Path("/home/fatemeh/Downloads/data_al_v1/test2/crops"), transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
"""