import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import tqdm
import wandb
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from tracking.association_learning import (AssDataset, AssociationNet,
                                           evaluate, save_model,
                                           train_one_epoch)

wandb.init(project="uncategorized")

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

main_dir = Path("/home/fatemeh/Downloads/data_al_v1")
save_path = Path("/home/fatemeh/Downloads/test_data")
exp = 2  # sys.argv[1]
no_epochs = 1  # int(sys.argv[2])


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_dataset = AssDataset(main_dir / "train/crops", transform=transform)
eval_dataset = AssDataset(main_dir / "valid/crops", transform=transform)
len_train, len_eval = len(train_dataset), len(eval_dataset)

"""
image_dir = Path("/home/fatemeh/Downloads/test_data/crops")
dataset = AssDataset(image_dir, transform=transform)
len_dataset = len(dataset)
len_train = int(len_dataset * 0.5)  # 0.8
len_eval = len_dataset - len_train
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:len_train])
eval_dataset = torch.utils.data.Subset(dataset, indices[len_train:])
"""

train_loader = DataLoader(
    train_dataset, batch_size=48, shuffle=True, num_workers=3, drop_last=False
)
eval_loader = DataLoader(
    eval_dataset, batch_size=48, shuffle=False, num_workers=1, drop_last=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = AssociationNet(512, 5).to(device)
# model.backbone.requires_grad_(False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1)

print(f"train: {len_train:,}, valid: {len_eval:,} images")
best_accuracy = 0
with tensorboard.SummaryWriter(save_path / f"tensorboard/{exp}") as writer:
    for epoch in tqdm.tqdm(range(1, no_epochs + 1)):
        training_time = int(time.time())
        train_one_epoch(
            train_loader, model, criterion, device, epoch, no_epochs, writer, optimizer
        )
        training_time = int(time.time()) - training_time
        print(f"training time: {training_time/60:.2f} min for {len_train:,} images")

        eval_time = int(time.time())
        accuracy = evaluate(
            eval_loader, model, criterion, device, epoch, no_epochs, writer
        )
        eval_time = int(time.time()) - eval_time
        print(f"evaluation time: {eval_time/60:.2f} min for {len_eval:,} images")
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            save_model(save_path, exp, epoch, model, optimizer, scheduler, best=True)
            print(f"best model accuracy: {best_accuracy:.2f} at epoch: {epoch}")
        # scheduler.step()

# 1-based save for epoch
save_model(save_path, exp, epoch, model, optimizer, scheduler)
