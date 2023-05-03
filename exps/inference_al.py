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

VISUALIZE = True
step = 8
frame_number = 64
time = 1
frame_number2 = frame_number + step * time
main_dir = Path("/home/fatemeh/Downloads/fish/data8_v3/train")
vid_name = "406_cam_1"
im = cv2.imread(f"{main_dir}/images/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
im2 = cv2.imread(f"{main_dir}/images/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets = da.get_detections_array(main_dir/f"labels/{vid_name}_frame_{frame_number:06d}.txt", im.shape[1], im.shape[0])
dets2 = da.get_detections_array(main_dir/f"labels/{vid_name}_frame_{frame_number2:06d}.txt", im.shape[1], im.shape[0])

device = 'cuda'
model = al.AssociationNet(512, 5).to(device)
model.load_state_dict(torch.load("/home/fatemeh/Downloads/result_snellius/al/1_best.pth")["model"])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
crop_w, crop_h = 512, 256
kdt = KDTree(dets2[:, 7:9])
_, inds = kdt.query(dets2[:,7:9], k=5)

if VISUALIZE:
    visualize.plot_detections_in_image(da.make_dets_from_array(dets), im[...,::-1]);plt.show(block=False)
    visualize.plot_detections_in_image(da.make_dets_from_array(dets2), im2[...,::-1]);plt.show(block=False)
    print(inds)

# for ind in inds:
#     for time in range(-4,5):
# vid_name = "04_07_22_F_2_rect_valid";frame_number=349; frame_number2=350;step=1
ind = [16, 11, 15, 18, 1]; time = 1 # small fish on top of large fish       80/100 {-4: 1, -3: 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
ind = [11, 16, 18, 15, 10]; time = 1 # big fish on behind the small fish    80/100 {-4: 1, -3: 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
ind = [10, 18, 11, 16, 15]; time = 1 # comouflage                           97/100
ind = [ 7,  6,  8,  9, 19]; time = 1  # similar fish around                 67/100
ind = [ 5,  4, 19,  3, 20]; time = 1  # similar fish around + comouflage    85/100 only one with jitter True is good and not the default


frame_number2 = frame_number + step * time
im2 = cv2.imread(f"{main_dir}/images/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets2 = da.get_detections_array(main_dir/f"labels/{vid_name}_frame_{frame_number2:06d}.txt", im.shape[1], im.shape[0])

# ind = [7, 1, 3, 4, 2]
def temporal_performance(jitter_x = 45, jitter_y = 65, jitter = False):
    count = dict(zip(np.arange(-4,5),np.arange(-4,5)*0))
    for time in range(-4,5):
        frame_number2 = frame_number + step * time
        im2 = cv2.imread(f"{main_dir}/images/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
        dets2 = da.get_detections_array(main_dir/f"labels/{vid_name}_frame_{frame_number2:06d}.txt", im.shape[1], im.shape[0]) # enter
        bbox1 = deepcopy(dets[ind[0], 2:7][None,:])
        if jitter:
            jitter_x, jitter_y = np.random.normal(50, 10, 2)
        crop_x, crop_y = max(0, int(bbox1[0, 1]+jitter_x-crop_w/2)), max(0, int(bbox1[0, 2]+jitter_y-crop_h/2))
        # crop_x, crop_y = max(0, int(bbox1[0, 1]-crop_w/2)), max(0, int(bbox1[0, 2]-crop_h/2))
        bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
        bboxes2 = deepcopy(dets2[ind, 2:7])
        bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y) # enter
        detc = bbox1.copy()
        detsc2 = bboxes2.copy() # enter
        bboxes2 = zero_out_of_image_bboxs(bboxes2, crop_w, crop_h)
        bbox1 = normalize_bboxs(bbox1, crop_w, crop_h)
        bboxes2 = normalize_bboxs(bboxes2, crop_w, crop_h) # enter
        bbox1 = torch.tensor(bbox1[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        bboxes2 = torch.tensor(bboxes2[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        time_emb = torch.tensor([time/4], dtype=torch.float32).to(device) # enter
        imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc = np.ascontiguousarray(imc)
        imc2 = np.ascontiguousarray(imc2)
        imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
        imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)
        output = model(imt, bbox1, imt2, bboxes2, time_emb)
        argmax = torch.argmax(output, axis=1).item()
        if argmax == 0:
            count[time] += 1
        print(frame_number2, ind, argmax, list(output.detach().cpu().numpy()[0]), int(jitter_x), int(jitter_y))
    print(count)
    visualize.plot_detections_in_image(detc, imc[...,::-1]);plt.show(block=False)
    visualize.plot_detections_in_image(detsc2, imc2[...,::-1]);plt.show(block=False)

def spatial_performance():
    count = 0
    for i in range(100):
        bbox1 = deepcopy(dets[ind[0], 2:7][None,:])
        jitter_x, jitter_y = np.random.normal(50, 10, 2)
        crop_x, crop_y = max(0, int(bbox1[0, 1]+jitter_x-crop_w/2)), max(0, int(bbox1[0, 2]+jitter_y-crop_h/2))
        # crop_x, crop_y = max(0, int(bbox1[0, 1]-crop_w/2)), max(0, int(bbox1[0, 2]-crop_h/2))
        bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
        bboxes2 = deepcopy(dets2[ind, 2:7])
        bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y) # enter
        detc = bbox1.copy()
        detsc2 = bboxes2.copy() # enter
        bboxes2 = zero_out_of_image_bboxs(bboxes2, crop_w, crop_h)
        bbox1 = normalize_bboxs(bbox1, crop_w, crop_h)
        bboxes2 = normalize_bboxs(bboxes2, crop_w, crop_h) # enter
        bbox1 = torch.tensor(bbox1[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        bboxes2 = torch.tensor(bboxes2[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        time_emb = torch.tensor([time/4], dtype=torch.float32).to(device) # enter
        imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc = np.ascontiguousarray(imc)
        imc2 = np.ascontiguousarray(imc2)
        imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
        imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)
        output = model(imt, bbox1, imt2, bboxes2, time_emb)
        argmax = torch.argmax(output, axis=1).item()
        if argmax == 0:
            count += 1
        print(frame_number2, ind, argmax, list(output.detach().cpu().numpy()[0]))
    print(count)
    visualize.plot_detections_in_image(detc, imc[...,::-1]);plt.show(block=False)
    visualize.plot_detections_in_image(detsc2, imc2[...,::-1]);plt.show(block=False)


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
time_emb = torch.tensor([1/4], dtype=torch.float32).to(device)

model(imt, bbox1, imt2, bboxes2, time_emb)

detsc = np.concatenate((dets[2:3, :3], dets[2:3, 3:9]-np.tile(np.array([crop_x, crop_y]),3), dets[2:3, 9:]), axis=1)
detsc2 = np.concatenate((dets2[ind, :3], dets2[ind, 3:9]-np.tile(np.array([crop_x, crop_y]),3), dets2[ind, 9:]), axis=1)
visualize.plot_detections_in_image(da.make_dets_from_array(detsc), imc[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(da.make_dets_from_array(detsc2), imc2[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(dets[:,2:7], im[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(dets2[:,2:7], im2[...,::-1]);plt.show(block=False)

dataset = al.AssDataset(Path("/home/fatemeh/Downloads/data_al_v1/test2/crops"), transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
item = next(iter(loader))
model(item[0].type(torch.float32).to(device),item[1].to(device), item[2].type(torch.float32).to(device), item[3].to(device),item[4].to(device))
"""

"""
image_path = Path("/home/fatemeh/Downloads/data_al_v1/test2/crops/04_07_22_F_2_rect_valid_349_350_1_1001_360.jpg")
image = cv2.imread(image_path.as_posix())[..., ::-1]
image = np.ascontiguousarray(image)
image1 = image[:256]
image2 = image[256:]

bbox_file = image_path.parent / f"{image_path.stem}.txt"
dets = np.loadtxt(bbox_file, skiprows=1, delimiter=",")

# normalize boxes: divide by image width
bboxs = dets.astype(np.float64).copy()
bboxs[:, 3:11:2] /= 512  # TODO
bboxs[:, 4:11:2] /= 256  # TODO
bbox1 = bboxs[0:1]
bboxs2 = bboxs[1:]
label = int(bboxs2[bboxs2[:, 0] == bbox1[0, 0], 2][0])
bbox1 = bbox1[:, [3, 4, 5, 6]].astype(np.float32)
bboxs2 = bboxs2[:, [3, 4, 5, 6]].astype(np.float32)
time = np.float32(int(image_path.stem.split("_")[-3]) / 4.0)  # TODO

image1 = transform(Image.fromarray(image1)).unsqueeze(0).to(device)
image2 = transform(Image.fromarray(image2)).unsqueeze(0).to(device)
bbox1 = torch.tensor(bbox1).unsqueeze(0).to(device)
bboxs2 = torch.tensor(bboxs2).unsqueeze(0).to(device)
time_emb = torch.tensor([time], dtype=torch.float32).to(device)
print(model(image1, bbox1, image2, bboxs2, time_emb))

visualize.plot_detections_in_image(bboxs[0:1,[0,3,4,5,6]], image1[...,::-1]);plt.show(block=False)
visualize.plot_detections_in_image(bboxs[1:,[0,3,4,5,6]], image2[...,::-1]);plt.show(block=False)
"""