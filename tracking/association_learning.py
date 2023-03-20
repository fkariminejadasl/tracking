from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import Bottleneck

import tracking.data_association as da


class TestDataset2(Dataset):
    def __init__(self, image_dir: Path, det_dir: Path, transform=None):
        self.image_files = list(image_dir.glob("*"))
        self.det_dir = det_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, ind):
        image_file = self.image_files[ind]
        image = cv2.imread(image_file.as_posix())[..., ::-1]
        image = np.ascontiguousarray(image)

        if self.transform:
            image = self.transform(image)

        bbox = np.random.rand(5, 4).astype(np.float32)
        label = np.arange(5)
        time = np.random.permutation(5)[0]
        result = {"image": image_file.stem, "time": time}
        print(image_file.stem)
        return result


class TestDataset(Dataset):
    def __init__(self, image_dir: Path, det_dir: Path, transform=None):
        self.req_height, self.req_width = 1080, 1920
        self.image_files = list(image_dir.glob("*"))
        self.det_dir = det_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, ind):
        image_file = self.image_files[ind]
        image = cv2.imread(image_file.as_posix())[..., ::-1]

        # crop to rquired size
        height, width, _ = image.shape
        height_cut = height - self.req_height
        width_cut1 = int((width - self.req_width) / 2)
        width_cut2 = self.req_width + width_cut1
        image = image[height_cut:, width_cut1:width_cut2, :]
        image = np.ascontiguousarray(image)
        # assert image.shape == (self.req_height, self.req_width, 3)
        # print(image_file)
        # print(image.shape)
        det_file = self.det_dir / f"{image_file.stem}.txt"
        dets = da.get_detections(det_file, width, height, zero_based=True)
        time = int(image_file.stem.split("_")[-1])
        # sample = {"image": image, "time": time, "label": dets}

        if self.transform:
            image = self.transform(image)

        # target = dict(image_id=torch.tensor(image), boxes=dets, labels=1)
        # return target
        return image, dets, 1


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
        # assert image.shape == (self.req_height, self.req_width, 3)
        print(image_path)
        print(image.shape)
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


image_dir = Path("/home/fatemeh/Downloads/test_data/crops")
det_dir = Path("/home/fatemeh/Downloads/test_data/labels")
# https://github.com/pytorch/vision/blob/main/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))


# testloader = torch.utils.data.DataLoader(test, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)


# dataset = al.AssDataset(al.image_dir)
# loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

"""
Superglue wise is that: give each image separately get features, and then do sinkhorn stuff (differentiable Hungarian). 
There is no location or time encoding. Image encode locations and time implicitly encoded in the other image.
"""

"""
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.conv1 = torch.nn.Conv2d(
    6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
model.fc = torch.nn.Linear(in_features=512, out_features=5, bias=True)


def load(checkpoint_path) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def save(checkpoint_path, model) -> None:
    torch.save(model.state_dict(), checkpoint_path)


writer = tensorboard.SummaryWriter(path / "tensorboard")
"""


class ConcatNet2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.image_fc = torch.nn.Linear(in_channels, 30)
        self.bbox_fc = torch.nn.Linear(5 * 4, 30)
        self.time_fc = torch.nn.Linear(1, 30)
        self.fc1 = torch.nn.Linear(90, 90)
        self.fc2 = torch.nn.Linear(90, 90)
        self.fc3 = torch.nn.Linear(90, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, emb: torch.Tensor, bbox: torch.Tensor, time: int):
        time = torch.tensor(time, dtype=torch.float32)
        time = time.repeat(emb.shape[0], 1)
        time = self.relu(self.time_fc(time))

        bbox = bbox.flatten().unsqueeze(0)
        bbox = self.relu(self.bbox_fc(bbox))

        emb = self.relu(self.image_fc(emb))
        x = torch.cat((emb, bbox, time), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AssociationNet2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        concat = ConcatNet(1000, 5)
        self.backbone = backbone
        self.conate = concat

    def forward(self, image: torch.Tensor, bbox: torch.Tensor, time: int):
        emb = self.backbone(image)
        emb = self.conate(emb, bbox, time)
        return emb

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
        time: int,
    ):
        time = torch.tensor(time, dtype=torch.float32)
        time = time.repeat(emb1.shape[0], 1)
        time = self.relu(self.time_fc(time))

        bbox1 = bbox1.flatten().unsqueeze(0)
        bbox1 = self.relu(self.bbox1_fc(bbox1))
        bboxs2 = bboxs2.flatten().unsqueeze(0)
        bboxs2 = self.relu(self.bbox2_fc(bboxs2))

        emb1 = self.relu(self.image_fc(emb1))
        emb2 = self.relu(self.image_fc(emb2))
        x = torch.cat((emb1, bbox1, emb2, bboxs2, time), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AssociationNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        concat = ConcatNet(1000, 5)
        self.backbone = backbone
        self.conate = concat

    def forward(self, image1: torch.Tensor, bbox1: torch.Tensor, image2: torch.Tensor, bboxs2: torch.Tensor, time: int):
        emb1 = self.backbone(image1)
        emb2 = self.backbone(image2)
        emb = self.conate(emb1, bbox1, emb2, bboxs2, time)
        return emb


def transform(x: np.ndarray) -> torch.Tensor:
    x = Image.fromarray(x)
    x = torchvision.transforms.functional.rotate(x, 30)
    x = torchvision.transforms.functional.to_grayscale(x)
    x = torchvision.transforms.functional.to_tensor(x)
    return x


def train_one_step():
    # get the inputs; data is a list of [inputs, labels]
    # inputs, labels = data
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    print(net)

    net.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    for epoch in range(2):  # loop over the dataset multiple times
        loss = train_one_step(data, device, net, criterion, optimizer, scheduler)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


"""
import cv2
import numpy as np
import torchvision

im = cv2.imread("/home/fatemeh/Downloads/vids/tttt.jpg")[..., ::-1]
im = np.ascontiguousarray(im)
imt = torchvision.transforms.functional.to_tensor(im).unsqueeze(0)
time = 5
bbox = torch.rand(5, 4)


associate = AssociationNet(1000, 5)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(associate.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, associate.parameters()), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

for epoch in range(2):  # loop over the dataset multiple times

    optimizer.zero_grad()

    # forward + backward + optimize
    # outputs = net(inputs)
    labels = torch.tensor(2).unsqueeze(0)  # Nx
    outputs = associate(imt, bbox, time)
    # net(tmp[0].permute((0,3,1,2)).type(torch.float32), tmp[3].squeeze()[:, 7:].type(torch.float32), int(tmp[4]))
    output = net(item[0].type(torch.float32), item[1].type(torch.float32), item[2].type(torch.float32), item[3].type(torch.float32), int(item[4]))

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss = 0.0
    running_loss += loss.item()
    print(running_loss, loss.item())

"""
