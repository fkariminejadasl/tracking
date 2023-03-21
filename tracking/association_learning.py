from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import Bottleneck

import tracking.data_association as da

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def denormalize_bboxs(bboxes, im_width, im_height):
    bboxs = bboxes.copy()
    bboxs[:, 3:11:2] *= im_width
    bboxs[:, 4:11:2] *= im_height
    return bboxs


class AssDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.image_paths = list(image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, ind):
        image_path = self.image_paths[ind]
        image = cv2.imread(image_path.as_posix())[..., ::-1]
        image1 = image[:256].transpose((2, 1, 0))
        image2 = image[256:].transpose((2, 1, 0))
        image1 = np.ascontiguousarray(image1)
        image2 = np.ascontiguousarray(image2)

        bbox_file = image_path.parent / f"{image_path.stem}.txt"
        bboxs = np.loadtxt(bbox_file, skiprows=1, delimiter=",").astype(np.float64)
        # normalize boxes: divide by image width
        bboxs[:, 3:11:2] /= 512  # TODO
        bboxs[:, 4:11:2] /= 256  # TODO
        bbox1 = bboxs[0:1]
        bboxs2 = bboxs[1:]
        label = int(bboxs2[bboxs2[:, 0] == bbox1[0, 0], 2][0])
        bbox1 = bbox1[:, [3, 4, 5, 6]]
        bboxs2 = bboxs2[:, [3, 4, 5, 6]]
        time = int(image_path.stem.split("_")[-3])
        # sample = {"image": image, "time": time, "label": dets}

        if self.transform:
            image = self.transform(image)

        # target = dict(image_id=torch.tensor(image), boxes=dets, labels=1)
        # return target
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
        time = time.unsqueeze(1).type(torch.float32)  # N -> Nx1
        time = self.relu(self.time_fc(time))

        bbox1 = bbox1.flatten(1).type(torch.float32)
        bbox1 = self.relu(self.bbox1_fc(bbox1))
        bboxs2 = bboxs2.flatten(1).type(torch.float32)
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
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
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
        time: int,
    ):
        emb1 = self.backbone(image1)
        emb2 = self.backbone(image2)
        emb = self.concat(emb1, bbox1, emb2, bboxs2, time)
        return emb


def transform(x: np.ndarray) -> torch.Tensor:
    x = Image.fromarray(x)
    x = torchvision.transforms.functional.rotate(x, 30)
    x = torchvision.transforms.functional.to_grayscale(x)
    x = torchvision.transforms.functional.to_tensor(x)
    return x


def train_one_epoch(loader, model, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0
    running_accuracy = 0
    print("train: ", len(loader.dataset))
    for i, item in enumerate(loader):
        optimizer.zero_grad()

        outputs = model(
            item[0].type(torch.float32).to(device),
            item[1].to(device),
            item[2].type(torch.float32).to(device),
            item[3].to(device),
            item[4].to(device),
        )  # N x C

        loss = criterion(outputs, item[5].to(device))  # 1
        loss.backward()
        optimizer.step()

        accuracy = (torch.argmax(output.data, 1) == item[-1]).sum().item()  #
        running_accuracy += accuracy
        running_loss += loss.item()

        # print(
        #     f"train: epoch: {epoch}, total loss: {loss.item()}, accuracy: {accuracy * 100/len(item[-1])}, no. correct: {accuracy}, bs:{len(item[-1])}"
        # )

    print(
        f"train: epoch: {epoch}, total loss: {running_loss/(i+1)}, accuracy: {running_accuracy /len(loader.dataset)* 100}, no. correct: {running_accuracy}, length data:{len(loader.dataset)}"
    )


@torch.no_grad()
def evaluate(loader, model, criterion, device, epoch):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    print("eval: ", len(loader.dataset))
    for i, item in enumerate(loader):
        outputs = model(
            item[0].type(torch.float32).to(device),
            item[1].to(device),
            item[2].type(torch.float32).to(device),
            item[3].to(device),
            item[4].to(device),
        )  # N x C

        loss = criterion(outputs, item[5].to(device))  # 1

        accuracy = (torch.argmax(output.data, 1) == item[-1]).sum().item()  #
        running_accuracy += accuracy
        running_loss += loss.item()
        # print(
        #     f"eval: epoch: {epoch}, total loss: {loss.item()}, accuracy: {accuracy * 100/len(item[-1])}, no. correct: {accuracy}, bs:{len(item[-1])}"
        # )

    print(
        f"eval: epoch: {epoch}, total loss: {running_loss/(i+1)}, accuracy: {running_accuracy /len(loader.dataset)* 100}, no. correct: {running_accuracy}, length data:{len(loader.dataset)}"
    )


"""
Superglue wise is that: give each image separately get features, and then do sinkhorn stuff (differentiable Hungarian). 
There is no location or time encoding. Image encode locations and time implicitly encoded in the other image.
"""


"""
# quick model test
im = cv2.imread("/home/fatemeh/Downloads/vids/tttt.jpg")[..., ::-1]
im = np.ascontiguousarray(im)
imt = torchvision.transforms.functional.to_tensor(im).unsqueeze(0)
time = 5
bbox = torch.rand(1, 4)
bboxs = torch.rand(5, 4)
net = AssociationNet(512, 2)
net(imt, torch.rand(1, 4), imt, bbox, time)
"""

image_dir = Path("/home/fatemeh/Downloads/test_data/crops")
dataset = AssDataset(image_dir)
len_dataset = len(dataset)
len_train = int(len_dataset * .8)
len_eval = len_dataset - len_train
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:len_train])
eval_dataset = torch.utils.data.Subset(dataset, indices[len_train:])

train_loader = DataLoader(
    train_dataset, batch_size=24, shuffle=False, num_workers=1, drop_last=False
)
eval_loader = DataLoader(
    eval_dataset, batch_size=8, shuffle=False, num_workers=1, drop_last=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AssociationNet(512, 5).to(device)
model.backbone.requires_grad_(False)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

print(len_train, len_eval)
print(len(train_loader), len(eval_loader))
print(len(train_loader.dataset), len(eval_loader.dataset))
for epoch in range(2):
    print("training")
    train_one_epoch(train_loader, model, optimizer, criterion, device, epoch)
    scheduler.step()
    print("evaluating")
    evaluate(eval_loader, model, criterion, device, epoch)


"""
def load(checkpoint_path) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def save(checkpoint_path, model) -> None:
    torch.save(model.state_dict(), checkpoint_path)
"""
