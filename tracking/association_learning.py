import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def denormalize_bboxs(bboxes: np.ndarray, im_width, im_height) -> np.ndarray:
    bboxs = bboxes.copy()
    bboxs[:, ::2] *= im_width
    bboxs[:, 1::2] *= im_height
    bboxs = np.hstack((np.arange(len(bboxs))[:, None], bboxs))
    return bboxs


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    image = image.numpy().transpose((1, 2, 0))
    image = 255 * (image - image.min()) / (image.max() - image.min())
    return image.astype(np.uint8)


class AssDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.image_paths = list(image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, ind):
        image_path = self.image_paths[ind]
        image = cv2.imread(image_path.as_posix())[..., ::-1]
        image = np.ascontiguousarray(image)
        image1 = image[:256]
        image2 = image[256:]

        bbox_file = image_path.parent / f"{image_path.stem}.txt"
        bboxs = np.loadtxt(bbox_file, skiprows=1, delimiter=",").astype(np.float64)
        # normalize boxes: divide by image width
        bboxs[:, 3:11:2] /= 512  # TODO
        bboxs[:, 4:11:2] /= 256  # TODO
        bbox1 = bboxs[0:1]
        bboxs2 = bboxs[1:]
        label = int(bboxs2[bboxs2[:, 0] == bbox1[0, 0], 2][0])
        bbox1 = bbox1[:, [3, 4, 5, 6]].astype(np.float32)
        bboxs2 = bboxs2[:, [3, 4, 5, 6]].astype(np.float32)
        time = np.float32(int(image_path.stem.split("_")[-3]) / 4.0)  # TODO

        if self.transform:
            image1 = self.transform(Image.fromarray(image1))
            image2 = self.transform(Image.fromarray(image2))
        else:
            image1 = image1.transpose((2, 1, 0))
            image2 = image2.transpose((2, 1, 0))

        return image1, bbox1, image2, bboxs2, time, label


class ConcatNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.image_fc = torch.nn.Linear(in_channels, 30)
        self.bbox1_fc = torch.nn.Linear(1 * 4, 30)
        self.bbox2_fc = torch.nn.Linear(5 * 4, 30)
        self.time_fc = torch.nn.Linear(1, 30)
        self.fc1 = torch.nn.Linear(150, 90)
        self.fc2 = torch.nn.Linear(90, 90)
        self.fc3 = torch.nn.Linear(90, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        emb1: torch.Tensor,
        bbox1: torch.Tensor,
        emb2: torch.Tensor,
        bboxs2: torch.Tensor,
        time: torch.Tensor,
    ):
        time = time.unsqueeze(1)  # N -> Nx1
        time = self.relu(self.time_fc(time))

        bbox1 = bbox1.flatten(1)
        bbox1 = self.relu(self.bbox1_fc(bbox1))
        bboxs2 = bboxs2.flatten(1)
        bboxs2 = self.relu(self.bbox2_fc(bboxs2))

        emb1 = self.relu(self.image_fc(emb1))
        emb2 = self.relu(self.image_fc(emb2))
        x = torch.cat((emb1, bbox1, emb2, bboxs2, time), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PartResnet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.part = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

    def forward(self, x):
        x = self.part(x).flatten(1)
        return x


class AssociationNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        backbone = PartResnet(resnet)

        concat = ConcatNet(in_channels, out_channels)
        self.backbone = backbone
        self.concat = concat

    def forward(
        self,
        image1: torch.Tensor,
        bbox1: torch.Tensor,
        image2: torch.Tensor,
        bboxs2: torch.Tensor,
        time: torch.Tensor,
    ):
        emb1 = self.backbone(image1)
        emb2 = self.backbone(image2)
        emb = self.concat(emb1, bbox1, emb2, bboxs2, time)
        return emb


def write_info_in_tensorboard(writer, epoch, loss, accuracy, stage):
    loss_scalar_dict = dict()
    acc_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    acc_scalar_dict[stage] = accuracy
    writer.add_scalars("loss", loss_scalar_dict, epoch)
    writer.add_scalars("accuracy", acc_scalar_dict, epoch)


def train_one_epoch(
    loader, model, criterion, device, epoch, no_epochs, writer, optimizer
):
    data_len = len(loader.dataset)
    batch_size = next(iter(loader))[-1].shape[0]
    no_batches = int(np.ceil(data_len / batch_size))
    print(f"train: number of batches: {no_batches:,}")
    start_time = time.time()

    model.train()
    running_loss = 0
    running_corrects = 0
    for i, item in enumerate(loader):
        optimizer.zero_grad()

        outputs = model(
            item[0].type(torch.float32).to(device),
            item[1].to(device),
            item[2].type(torch.float32).to(device),
            item[3].to(device),
            item[4].to(device),
        )  # N x C

        labels = item[5].to(device)
        loss = criterion(outputs, labels)  # 1
        loss.backward()
        optimizer.step()

        corrects = (torch.argmax(outputs.data, 1) == labels).sum().item()
        running_corrects += corrects
        running_loss += loss.item()

        if i % 100 == 0 and i != 0:
            print(f"{i}, {time.time()-start_time:.1f}")
            print(
                f"train: current/total: {i+1}/{no_batches}, total loss: {loss.item():.4f}, accuracy: {corrects * 100/batch_size:.2f}, no. correct: {corrects}, bs:{batch_size}"
            )
            start_time = time.time()

    total_loss = running_loss / (i + 1)
    total_accuracy = running_corrects / data_len * 100
    print(
        f"train: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}, accuracy: {total_accuracy:.2f}, no. correct: {running_corrects}, length data:{len(loader.dataset)}"
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage="train")


@torch.no_grad()
def evaluate(loader, model, criterion, device, epoch, no_epochs, writer):
    data_len = len(loader.dataset)
    batch_size = next(iter(loader))[-1].shape[0]
    no_batches = int(np.ceil(data_len / batch_size))
    model.eval()
    running_loss = 0
    running_corrects = 0
    for i, item in enumerate(loader):
        outputs = model(
            item[0].type(torch.float32).to(device),
            item[1].to(device),
            item[2].type(torch.float32).to(device),
            item[3].to(device),
            item[4].to(device),
        )  # N x C

        labels = item[5].to(device)
        loss = criterion(outputs, labels)  # 1

        corrects = (torch.argmax(outputs.data, 1) == labels).sum().item()
        running_corrects += corrects
        running_loss += loss.item()

        # print(
        #     f"eval: current/total: {i+1}/{no_batches}, total loss: {loss.item():.4f}, accuracy: {corrects * 100/batch_size:.2f}, no. correct: {corrects}, bs:{batch_size}"
        # )

    total_loss = running_loss / (i + 1)
    total_accuracy = running_corrects / data_len * 100
    print(
        f"eval: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}, accuracy: {total_accuracy:.2f}, no. correct: {running_corrects}, length data:{len(loader.dataset)}"
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage="valid")
    return total_accuracy


def load_model(checkpoint_path, model, device) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    return model


def save_model(
    checkpoint_path, exp, epoch, model, optimizer, scheduler, best=False
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "date": datetime.now().isoformat(),
    }
    name = f"{exp}_{epoch}.pth"
    if best:
        name = f"{exp}_best.pth"
    torch.save(checkpoint, checkpoint_path / name)


"""
Superglue wise is that: give each image separately get features, and then do sinkhorn stuff (differentiable Hungarian). 
There is no location or time encoding. Image encode locations and time implicitly encoded in the other image.
"""

"""
# quick model test
im = torch.zeros((1, 3, 512, 256), dtype=torch.float32)
bbox = torch.zeros((1, 1, 4), dtype=torch.float32)
bboxs = torch.zeros((1, 5, 4), dtype=torch.float32)
time = torch.tensor([0]).to(torch.float32)
net = AssociationNet(512, 5)
net(im, bbox, im, bboxs, time)
"""
