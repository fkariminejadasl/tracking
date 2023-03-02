import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import Bottleneck
from pathlib import Path
import cv2


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


image_dir = Path("/home/fatemeh/Downloads/test_data/images")
det_dir = Path("/home/fatemeh/Downloads/test_data/labels")
# https://github.com/pytorch/vision/blob/main/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))


# testloader = torch.utils.data.DataLoader(test, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

# tetest_dataset = TestDataset(image_dir, det_dir)
# test_loader = DataLoader(
#         test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
#     )


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


class ConcatNet(torch.nn.Module):
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


class AssociationNet(torch.nn.Module):
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

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss = 0.0
    running_loss += loss.item()
    print(running_loss, loss.item())

"""
