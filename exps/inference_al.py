from copy import deepcopy
import cv2
from PIL import Image
import torch, torchvision
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KDTree
from tracking import association_learning as al, visualize, data_association as da
import time as ttime
from tqdm import tqdm

def normalize_bboxs(bboxs, crop_w, crop_h):
    assert bboxs.shape[1] == 5
    bboxs = bboxs.astype(np.float32)
    return np.concatenate((bboxs[:, 0:1], bboxs[:, 1:]/np.tile(np.array([crop_w, crop_h]), 2)), axis=1)


def change_center_bboxs(bboxs, crop_x, crop_y):
    assert bboxs.shape[1] == 5
    return np.concatenate((bboxs[:, 0:1], bboxs[:, 1:]-np.tile(np.array([crop_x, crop_y]),2)), axis=1)

def zero_out_of_image_bboxs2(bboxs, crop_w, crop_h):
    assert bboxs.shape[1] == 5
    inds = np.where((bboxs[:,1]>=crop_w) | (bboxs[:,1]<0)| (bboxs[:,3]>=crop_w) | (bboxs[:,3]<0) | (bboxs[:,2]>=crop_h) | (bboxs[:,2]<0) | (bboxs[:,4]>=crop_h) | (bboxs[:,4]<0))[0]
    bboxs[inds, 1:] = 0
    return bboxs

def zero_out_of_image_bboxs(bboxes, crop_w, crop_h):
    assert bboxes.shape[1] == 5
    bboxs = bboxes.copy()
    bboxs[:, 1::2] = np.clip(bboxs[:,1::2], 0, crop_w)
    bboxs[:, 2::2] = np.clip(bboxs[:,2::2], 0, crop_h)
    bboxs[bboxs[:, 1] == bboxs[:, 3], 1:] = 0
    bboxs[bboxs[:, 2] == bboxs[:, 4], 1:] = 0
    return bboxs

def test_zero_out_of_image_bboxs():
    bboxs = np.array([
       [  9, 205,  -1, 216,  27],
       [ 10, 489,  30, 501,  49],
       [ 11, 406, 427, 417, 434],
       [  5, 416, 548, 453, 570],
       [  6, 541, 515, 558, 529]])
    desired = np.array([
       [  9, 205,   0, 216,  27],
       [ 10, 489,  30, 501,  49],
       [ 11,   0,   0,   0,   0],
       [  5,   0,   0,   0,   0],
       [  6,   0,   0,   0,   0]])
    actual = zero_out_of_image_bboxs(bboxs, 512, 256)
    np.testing.assert_equal(actual, desired)

def zero_padding_images(image, crop_w, crop_h):
    pad_x = max(0, crop_w - image.shape[1])
    pad_y = max(0, crop_h - image.shape[0])
    image = np.pad(image, ((0, pad_y), (0, pad_x), (0,0)))
    return image

def save_overview_images_dets(overview_dir: Path, name_stem: str, image1, bbox1, image2, bboxs2, crop_h):
    assert bbox1.shape[1] == 5
    assert bboxs2.shape[1] == 5
    if not overview_dir.exists():
        overview_dir.mkdir(parents=True, exist_ok=True)
    bboxs2_shift = bboxs2.copy()
    # hack: remove the negative y-values (out of image), since after adding crop_h they come to the first image
    bboxs2_shift = bboxs2_shift[np.max(bboxs2_shift[:,[2,4]],axis=1)>=0]
    bboxs2_shift[:, [2, 4]] = bboxs2_shift[:, [2, 4]] + crop_h
    bboxs12 = np.concatenate((bbox1, bboxs2_shift), axis=0)
    image12 = np.concatenate((image1, image2), axis=0)
    visualize.plot_detections_in_image(bboxs12, image12)
    fig = plt.gcf()
    fig.set_figwidth(4.8)
    fig.savefig(overview_dir / f"{name_stem}.jpg")
    plt.close()

def spatial_performance(query_ind, ind, im, dets, im2, dets2, time, frame_number, frame_number2, VISUALIZE=False):
    #crop_w, crop_h, device, transform, model, VISUALIZE=False):
    query_track_id = dets[query_ind, 0]
    track_ids = dets2[ind, 0]
    if query_track_id in track_ids:
        label = np.where(track_ids==query_track_id)[0][0]
    else:
        return -1, np.empty((0, 5), dtype=np.int64)

    count = 0
    jitters = []
    for i in range(10): #100
        # bbox1 = deepcopy(dets[query_ind, 2:7][None,:])
        bbox1 = deepcopy(dets[query_ind, [0,3,4,5,6]][None,:])
        jitter_x, jitter_y = np.random.normal(50, 10, 2)
        crop_x, crop_y = max(0, int(bbox1[0, 1]+jitter_x-crop_w/2)), max(0, int(bbox1[0, 2]+jitter_y-crop_h/2))
        bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
        # bboxes2 = deepcopy(dets2[ind, 2:7])
        bboxes2 = deepcopy(dets2[ind][:,[0,3,4,5,6]])
        bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y) 
        
        detc = bbox1.copy()
        detsc2 = bboxes2.copy() 
        
        bboxes2 = zero_out_of_image_bboxs(bboxes2, crop_w, crop_h)
        bbox1 = normalize_bboxs(bbox1, crop_w, crop_h)
        bboxes2 = normalize_bboxs(bboxes2, crop_w, crop_h) 
        
        bbox1 = torch.tensor(bbox1[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        bboxes2 = torch.tensor(bboxes2[:,1:]).unsqueeze(0).to(device).to(torch.float32)
        time_emb = torch.tensor([time/4], dtype=torch.float32).to(device) 
        
        imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        imc = np.ascontiguousarray(imc)
        imc2 = np.ascontiguousarray(imc2)
        imt = transform(Image.fromarray(imc)).unsqueeze(0).to(device)
        imt2 = transform(Image.fromarray(imc2)).unsqueeze(0).to(device)
        output = model(imt, bbox1, imt2, bboxes2, time_emb)
        argmax = torch.argmax(output, axis=1).item()

        track_ids = dets2[ind][:,0]

        if argmax == label:
            count += 1
            jitters.append([jitter_x, jitter_y, 1, track_ids[label], track_ids[argmax]])
        else:
            jitters.append([jitter_x, jitter_y, 0, track_ids[label], track_ids[argmax]])
    # print(count)
    if VISUALIZE:
        visualize.plot_detections_in_image(detc, imc[...,::-1]);plt.show(block=False)
        visualize.plot_detections_in_image(detsc2, imc2[...,::-1]);plt.show(block=False)
    return count, np.array(jitters).astype(np.int64)

def save_associations(filename):
    start_time = ttime.time()
    time, start_frame, end_frame, step = 1, 0, 256, 8
    with open(filename, 'w') as wfile:
        wfile.write(f"vid_name, frame_number, frame_number2, time, query_track_id, track_ids, correct perc, query_ind, inds, loc_x, loc_y, correctness, label, pred\n")
        for track_file in track_dir.glob("*.zip"):
            vid_name = track_file.stem
            tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
            for frame_number in tqdm(range(start_frame, end_frame, step)):
                frame_number2 = frame_number + step * time
                im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
                im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
                dets = tracks[tracks[:,1]==frame_number].copy()
                dets2 = tracks[tracks[:,1]==frame_number2].copy()
                kdt = KDTree(dets2[:, 7:9])
                _, inds = kdt.query(dets[:,7:9], k=5)
                for query_ind, ind in enumerate(inds):
                    query_track_id = dets[query_ind,0]
                    count, jitters = spatial_performance(query_ind, ind, im, dets, im2, dets2, time, frame_number, frame_number2)
                    if count !=-1:
                        for jitter in jitters:
                            wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{query_track_id},{','.join([str(i[0]) for i in dets2[ind]])},{count},{query_ind},{','.join([str(i) for i in ind])},{jitter[0]+dets[query_ind,3]},{jitter[1]+dets[query_ind,4]},{jitter[2]},{jitter[3]},{jitter[4]}\n")
                    else:
                        wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{query_track_id},{','.join([str(i[0]) for i in dets2[ind]])},-1,{query_ind},{','.join([str(i) for i in ind])}\n")
    end_time = ttime.time()
    print(f"total time: {int(end_time - start_time)}")

def get_saved_results(filename):
    results = []
    with open(filename, 'r') as rfile:
        rfile.readline()
        for row in rfile:
            items = row.split("\n")[0].split(',')
            count = int(items[10])
            if count !=-1:
                vid_name = int(items[0].split('_')[0])
                frame_number = int(items[1])
                frame_number2 = int(items[2])
                time = int(items[3])
                query_track_id = int(items[4])
                track_ids = [int(i) for i in items[5:10]]
                query_ind = int(items[11])
                ind = [int(i) for i in items[12:17]]
                pos_x = int(items[17])
                pos_y = int(items[18])
                success = int(items[19])
                label = int(items[20])
                pred = int(items[21])
                results.append([vid_name, frame_number, frame_number2, time, query_track_id, *track_ids, count, query_ind, *ind, pos_x, pos_y, success, label, pred])
    results = np.array(results)
    return results

def save_overview_images_dets_for_bad_results(results, overview_dir):
    track_files = list(track_dir.glob("*.zip"))
    for track_file in tqdm(track_files):
        vid_name = track_file.stem
        tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
        vid_name_number = int(vid_name.split('_')[0])
        # select failed cases based on video name, count, failure
        failed = results[(results[:,0] == vid_name_number) & (results[:,10]<6) & (results[:,19]==0)]
        # select only one example from failed cases, based on frame number and track id
        _, failed_inds = np.unique(failed[:,[1,4]], axis=0, return_index=True)

        for ind in failed_inds:
            frame_number, frame_number2, query_track_id, count = failed[ind,1], failed[ind,2], failed[ind,4], failed[ind,10]
            query_ind, other_inds = failed[ind,11], failed[ind,12:17]
            pos_x, pos_y, label, pred = failed[ind,17], failed[ind,18], failed[ind,20], failed[ind,21]

            im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")
            im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")
            dets = tracks[tracks[:,1]==frame_number].copy()
            dets2 = tracks[tracks[:,1]==frame_number2].copy()

            bbox1 = deepcopy(dets[query_ind, [0,3,4,5,6]][None,:])
            crop_x, crop_y = max(0, int(pos_x-crop_w/2)), max(0, int(pos_y-crop_h/2))
            bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
            bboxs2 = deepcopy(dets2[other_inds][:,[0,3,4,5,6]])
            bboxs2 = change_center_bboxs(bboxs2, crop_x, crop_y) 
            
            detc = bbox1.copy()
            detsc2 = bboxs2.copy()
            imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

            name_stem = f"{vid_name_number}_{frame_number}_{label}_{pred}_{count}_{pos_x}_{pos_y}_{query_track_id}_{'_'.join([str(i) for i in bboxs2[:, 0]])}" 
            save_overview_images_dets(overview_dir, name_stem, imc, detc, imc2, detsc2, crop_h)

device = 'cuda'
# model = al.AssociationNet(512, 5).to(device)
# model.load_state_dict(torch.load("/home/fatemeh/Downloads/result_snellius/al/1_best.pth")["model"])
model = al.AssociationNet(2048, 5).to(device)
model.load_state_dict(torch.load("/home/fatemeh/Downloads/result_snellius/al/2_best.pth")["model"])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
crop_w, crop_h = 512, 256

video_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/vids")
track_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/mots")
image_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/images")
overview_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/overview")

# save_associations("/home/fatemeh/Downloads/10_jitters_resnet18_2.txt")
# save_associations("/home/fatemeh/Downloads/10_jitters_resnet50_2.txt")
# results1 = get_saved_results("/home/fatemeh/Downloads/10_jitters_resnet18_1.txt")
# results2 = get_saved_results("/home/fatemeh/Downloads/10_jitters_resnet50_2.txt")
# save_overview_images_dets_for_bad_results(results1, overview_dir/"less_6_resnet50")

# results[(results[:,10]<6) & (results[:,19] == 0) & (results[:,0]==234)& (results[:,1]==8) & (results[:,20]==50)][:,20:]# & (results[:,21]==51) & (results[:,17]==1012) & (results[:,18]==73)]
# results = results2.copy()
# failed = results[(results[:,10]<6)]
# for vid_name_number in np.unique(failed[:,0]):
#     item1 = sum(failed[:,0]==vid_name_number)
#     item2 = sum(results[:,0]==vid_name_number)
#     print(f"{vid_name_number:3d}: {int(item1/10):3d}, {int(item2/10):4d}")
'''
# analysis 3 to save bad results
vid_names = []
results = []
with open("/home/fatemeh/Downloads/all_100_jitters.txt", 'r') as rfile:
    rfile.readline()
    for row in rfile:
        items = row.split("\n")[0].split(',')
        vid_names.append(items[0])
        frame_number = int(items[1])
        frame_number2 = int(items[2])
        time = int(items[3])
        track_id = int(items[4])
        count = int(items[5])
        query_ind = int(items[6])
        ind = [int(i) for i in items[7:]]
        results.append([frame_number, frame_number2, time, track_id, count, query_ind, *ind])
results = np.array(results)

inds = np.where(results[:,4]<60)[0]
# vid_name = '247_cam12'
# frames_tracks = np.array([(results[ind,0], results[ind,3]) for ind in inds if vid_names[ind]==vid_name])
# tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")

for track_file in track_dir.glob("*.zip"):
    vid_name = track_file.stem
    tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
    for ind in inds:
        if vid_names[ind] == vid_name:
            frame_number, frame_number2, query_track_id, count = results[ind,0], results[ind,1], results[ind,3], results[ind,4]
            query_ind, other_inds = results[ind,5], results[ind,6:]
            im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")
            im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")
            dets = tracks[tracks[:,1]==frame_number].copy()
            dets2 = tracks[tracks[:,1]==frame_number2].copy()

            bbox1 = deepcopy(dets[query_ind, [0,3,4,5,6]][None,:])
            jitter_x, jitter_y = np.random.normal(50, 10, 2)
            crop_x, crop_y = max(0, int(bbox1[0, 1]+jitter_x-crop_w/2)), max(0, int(bbox1[0, 2]+jitter_y-crop_h/2))
            bbox1 = change_center_bboxs(bbox1, crop_x, crop_y)
            bboxes2 = deepcopy(dets2[other_inds][:,[0,3,4,5,6]])
            bboxes2 = change_center_bboxs(bboxes2, crop_x, crop_y) # enter
            detc = bbox1.copy()
            detsc2 = bboxes2.copy()
            imc = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            imc2 = im2[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

            name_stem = f"{vid_name}_{frame_number}_{count}_{query_track_id}_{'_'.join([str(i) for i in bboxes2[:, 0]])}_{dets[query_ind,3]}_{dets[query_ind,4]}" 
            save_overview_images_dets(overview_dir, name_stem, imc, detc, imc2, detsc2, crop_h)
'''

""""
plt.figure()
x_y_counts = []
for ind in inds:
    if vid_names[ind] == vid_name:
        frame_number, frame_number2, query_track_id, count = results[ind,0], results[ind,1], results[ind,3], results[ind,4]
        query_ind, other_inds = results[ind,5], results[ind,6:]
        xtl, ytl, xbr, ybr, _, _, w, h = tracks[(tracks[:,0]==query_track_id)&(tracks[:,1]==frame_number)][0,3:]
        x_y_counts.append([xtl, ytl, xbr, ybr, w, h, count])
"""

"""
VISUALIZE = False
video_dir = Path("/home/fatemeh/Downloads/fish/vids/all")
track_dir = Path("/home/fatemeh/Downloads/fish/vids/mot")
image_dir = Path("/home/fatemeh/Downloads/fish/data8_v3/train/images")
vid_name = "406_cam_1"
step = 8
tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")


frame_number = 2344# 64, 2318, 2338
time = 1
frame_number2 = frame_number + step * time
im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets = tracks[tracks[:,1]==frame_number].copy()
dets2 = tracks[tracks[:,1]==frame_number2].copy()

kdt = KDTree(dets2[:, 7:9])
_, inds = kdt.query(dets2[:,7:9], k=5)

if VISUALIZE:
    print(inds)
    visualize.plot_detections_in_image(dets[:,[0,3,4,5,6]], im[...,::-1]);plt.show(block=False)
    visualize.plot_detections_in_image(dets2[:,[0,3,4,5,6]], im2[...,::-1]);plt.show(block=False)
    
# # vid_name = "04_07_22_F_2_rect_valid";frame_number=349; frame_number2=350;step=1
# ind = [16, 11, 15, 18, 1]; time = 1 # small fish on top of large fish       80/100 {-4: 1, -3: 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
# ind = [11, 16, 18, 15, 10]; time = 1 # big fish on behind the small fish    80/100 {-4: 1, -3: 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
# ind = [10, 18, 11, 16, 15]; time = 1 # comouflage                           97/100
# ind = [ 7,  6,  8,  9, 19]; time = 1  # similar fish around                 67/100
# ind = [ 5,  4, 19,  3, 20]; time = 1  # similar fish around + comouflage    85/100 only one with jitter True is good and not the default

# plt.figure();plt.plot(jitters[jitters[:,2]==0, 0], jitters[jitters[:,2]==0, 1], 'bo');plt.plot(jitters[jitters[:,2]==1, 0], jitters[jitters[:,2]==1, 1], 'r*');plt.show(block=False)

# fish_jitters = np.empty((0, 4), dtype=np.float32)
# for ind in inds:
#     count, jitters = spatial_performance()
#     fish_jitters = np.concatenate((jitters, fish_jitters),axis=0)


### analyses
vid_names = []
results = []
with open("/home/fatemeh/Downloads/all_100_jitters.txt", 'r') as rfile:
    rfile.readline()
    for row in rfile:
        items = row.split("\n")[0].split(',')
        vid_names.append(items[0])
        frame_number = int(items[1])
        frame_number2 = int(items[2])
        time = int(items[3])
        track_id = int(items[4])
        count = int(items[5])
        query_ind = int(items[6])
        ind = [int(i) for i in items[7:]]
        results.append([frame_number, frame_number2, time, track_id, count, query_ind, *ind])
results = np.array(results)

# 1. histogram of the jittering counts
plt.figure();histbins = plt.hist(results[:,4],bins=range(101));plt.title("histogram of the jitter counts");plt.show(block=False)
sum(results[:,4]>60)/len(results)
np.concatenate((np.int64(histbins[1])[1:-3][:, None],np.int64(histbins[0])[1:-2][:, None]), axis=1)

# 2. zero-counts and not detection
inds = np.where(results[:,4]==0)[0]
vid_names_uniq_zero = {vid_names[ind] for ind in inds}
hist_zeros = dict(zip(vid_names_uniq_zero, len(vid_names_uniq_zero)*[0]))
for vid_name in vid_names_uniq_zero:
    for ind in inds: 
        if vid_names[ind]==vid_name:
            hist_zeros[vid_name] += 1
            print((vid_names[ind], results[ind,0], results[ind,3]))
dict(sorted(hist_zeros.items(), key=lambda x:x[1]))

vid_name = '247_cam12'
frames_tracks = np.array([(results[ind,0], results[ind,3]) for ind in inds if vid_names[ind]==vid_name])
tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")

frames_tracks[np.argsort(frames_tracks[:,1])]

tracks[(tracks[:,0]==59)&(tracks[:,1]==248)]

[(vid_names[ind], results[ind,0], results[ind,3]) for ind in inds if vid_names[ind]=='247_cam12']
ind = [ind for ind in inds if (vid_names[ind]=='378_cam12') and (results[ind,0]==208) and (results[ind,3]==13)][0] #31719 


import inference_al as b
video_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/vids")
track_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/mots")
image_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/images")
overview_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/overview")
vid_name = vid_names[ind]
frame_number, frame_number2 = results[ind,0], results[ind,1]
tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets = tracks[tracks[:,1]==frame_number].copy()
dets2 = tracks[tracks[:,1]==frame_number2].copy()
count, jitters = b.spatial_performance(results[ind,5], results[ind,6:], im, dets, im2, dets2, 1, frame_number, frame_number2, True)

>>> ind = [4,3,2,5,1]
>>> frame_number, frame_number2, time, track_id = 256,264,1,4
>>> im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
>>> im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
>>> dets = tracks[tracks[:,1]==frame_number].copy()
>>> dets2 = tracks[tracks[:,1]==frame_number2].copy()
>>> visualize.plot_detections_in_image(dets[:,[0,3,4,5,6]], im[...,::-1]);plt.show(block=False)
>>> count, jitters = spatial_performance(ind, im, dets, im2, dets2, time, frame_number, frame_number2, True)
>>> dets2[ind, 0]
>>> dets[ind[0], 0]



with open("/home/fatemeh/Downloads/tmp.txt", 'w') as wfile:
    wfile.write(f"vid_name, frame_number, frame_number2, time, track_id, correct perc, ind\n")
    for ind in inds:
        count, jitters = spatial_performance(ind, im, dets, im2, dets2, time, frame_number, frame_number2)
        if count < 60:
            wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{dets[ind[0],0]},{count},{','.join([str(i) for i in ind])}\n")

import time as ttime
start_time = ttime.time()
frame_number = 2344# 64, 2318, 2338
im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
dets = tracks[tracks[:,1]==frame_number].copy()
kdt = KDTree(dets[:, 7:9])
_, inds = kdt.query(dets[:,7:9], k=5)
with open("/home/fatemeh/Downloads/tmp.txt", 'w') as wfile:
    wfile.write(f"vid_name, frame_number, frame_number2, time, track_id, correct perc, ind\n")
    for time in range(-4, 5):
        frame_number2 = frame_number + step * time
        im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
        dets2 = tracks[tracks[:,1]==frame_number2].copy()
        for ind in inds: # [[3,2,4,5,1]]
            count, jitters = spatial_performance(ind, im, dets, im2, dets2, time, frame_number, frame_number2)
            if count < 60:
                wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{dets[ind[0],0]},{count},{','.join([str(i) for i in ind])}\n")
end_time = ttime.time()
print(f"total time: {int(end_time - start_time)}")

                

start_time = ttime.time()
time = 1
with open("/home/fatemeh/Downloads/tmp.txt", 'w') as wfile:
    wfile.write(f"vid_name, frame_number, frame_number2, time, track_id, correct perc, ind\n")
    for frame_number in range(0, 3118, 32):
        frame_number2 = frame_number + step * time
        im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
        im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
        dets = tracks[tracks[:,1]==frame_number].copy()
        dets2 = tracks[tracks[:,1]==frame_number2].copy()
        kdt = KDTree(dets[:, 7:9])
        _, inds = kdt.query(dets[:,7:9], k=5)
        for ind in inds:
            count, jitters = spatial_performance(ind, im, dets, im2, dets2, time, frame_number, frame_number2)
            wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{dets[ind[0],0]},{count},{','.join([str(i) for i in ind])}\n")
end_time = ttime.time()
print(f"total time: {int(end_time - start_time)}")


visualize.save_video_as_images(video_file, save_path, 32)

for frame_number in range(0,3112,512):
        im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
        dets = tracks[tracks[:,1]==frame_number].copy()
        visualize.plot_detections_in_image(dets[:,[0,3,4,5,6]], im[...,::-1]);plt.show(block=False)

for frame_number in range(2344 - 8 * 4,2344 + 8 * 5, 8):
        im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
        dets = tracks[tracks[:,1]==frame_number].copy()
        visualize.plot_detections_in_image(dets[:,[0,3,4,5,6]], im[...,::-1]);plt.show(block=False)


video_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/vids")
track_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/mots")
image_dir = Path("/home/fatemeh/Downloads/fish/out_of_sample_vids_vids/images")

import time as ttime
start_time = ttime.time()
time = 1
start_frame, end_frame, step = 0, 256, 8

vid_name = "16_cam12"
tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
with open("/home/fatemeh/Downloads/tmp.txt", 'w') as wfile:
    wfile.write(f"vid_name, frame_number, frame_number2, time, track_id, correct perc, ind\n")
    for frame_number in range(start_frame, end_frame + step, step):
        frame_number2 = frame_number + step * time
im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
dets = tracks[tracks[:,1]==frame_number].copy()
dets2 = tracks[tracks[:,1]==frame_number2].copy()
kdt = KDTree(dets[:, 7:9])
_, inds = kdt.query(dets[:,7:9], k=5)
for ind in inds:
    count, jitters = spatial_performance(ind, im, dets, im2, dets2, time, frame_number, frame_number2)
            wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{dets[ind[0],0]},{count},{','.join([str(i) for i in ind])}\n")
end_time = ttime.time()
print(f"total time: {int(end_time - start_time)}")


end_frame = 260
for video_file in video_dir.glob("*.mp4"):
    vid_name = video_file.stem
    tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
    # visualize.save_video_with_tracks_as_images(tracks, video_file, image_dir, 8, end_frame=end_frame)
    visualize.save_video_as_images(video_file, image_dir, 8, end_frame=end_frame)
    print(f"vidname: {vid_name}, number of tracks: {len(np.unique(tracks[:,0]))}")
"""

"""    
start_time = ttime.time()

time, start_frame, end_frame, step = 1, 0, 256, 8
vid_name = "236_cam12"#16_cam12"
tracks = da.load_tracks_from_mot_format(track_dir / f"{vid_name}.zip")
with open(f"/home/fatemeh/Downloads/{vid_name}.txt", 'w') as wfile:
    wfile.write(f"vid_name, frame_number, frame_number2, time, track_id, correct perc, ind\n")
    for frame_number in tqdm(range(start_frame, end_frame, step)):
        frame_number2 = frame_number + step * time
        im = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number:06d}.jpg")[:,:,::-1]
        im2 = cv2.imread(f"{image_dir}/{vid_name}_frame_{frame_number2:06d}.jpg")[:,:,::-1]
        dets = tracks[tracks[:,1]==frame_number].copy()
        dets2 = tracks[tracks[:,1]==frame_number2].copy()
        kdt = KDTree(dets2[:, 7:9])
        _, inds = kdt.query(dets[:,7:9], k=5)
        for query_ind, ind in enumerate(inds):
            query_track_id = dets[query_ind,0]
            count, jitters = b.spatial_performance(query_ind, ind, im, dets, im2, dets2, time, frame_number, frame_number2)
            if count !=-1:
                wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{query_track_id},{count},{query_ind},{','.join([str(i) for i in ind])}\n")
            else:
                wfile.write(f"{vid_name},{frame_number},{frame_number2},{time},{query_track_id},-1,{query_ind},{','.join([str(i) for i in ind])}\n")
end_time = ttime.time()
print(f"total time: {int(end_time - start_time)}")
"""



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