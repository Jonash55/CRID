import os
import time
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from itertools import product

from torch import nn

from torch.utils.tensorboard import SummaryWriter

from utils.validation import validate
from utils.matplotlib_img_show import matplotlib_imshow
from utils.load_split_train_test import load_split_train_test


# dataset import
dataset_path = "../400_4_classes"

# splitting dataset into train and test sets,
# creating data loaders for each set
train_loader, test_loader = load_split_train_test(
    dataset_path, test_size=0.1, batch_size=16
)
classes = train_loader.dataset.classes

# settings training device to cuda (GPU), else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
        )

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 28 * 37, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = ConvNet().to(device)
# settings up SummaryWriter to log scores to TensorBoard
date = time.strftime("%Y%m%d-%H%M%S")
log_path = f"runs/convnet_graph_images_date{date}"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
writer = SummaryWriter(log_path)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# loading images and labels to GPU
images, labels = images.to(device), labels.to(device)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid)

# write to tensorboard
writer.add_image("trash_images", img_grid)

# visualizing model
writer.add_graph(model, images)
writer.close()
# to run tensorboard: < tensorboard --logdir=runs >


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# HYPERPARAMETER TUNING
parameters = dict(lr=[0.001], batch_size=[80, 64, 32])

param_values = [v for v in parameters.values()]
print(param_values)

for lr, batch_size in product(*param_values):
    print(lr, batch_size)

num_epochs = 30

for run_id, (lr, batch_size) in enumerate(product(*param_values)):
    print("run_id:", run_id + 1)

    model = ConvNet().to(device)
    train_loader, test_loader = load_split_train_test(dataset_path, 0.1, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    date = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"runs/convnet_exp_{run_id}_batch_size{batch_size}_lr{lr}_date{date}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    comment = f" batch_size = {batch_size} lr = {lr}"
    writer = SummaryWriter(log_path, comment=comment)
    writer = SummaryWriter(log_path)

    total_step = len(train_loader)
    running_loss = 0.0
    correct = 0
    iteration = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # getting inputs
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            running_loss += loss.item()

        writer.add_scalar("Loss", running_loss, epoch)
        writer.add_scalar("Accuracy", correct / labels.size(0) * 100, epoch)
        print(
            "epoch:",
            epoch,
            "/",
            num_epochs,
            "accuracy:",
            correct / labels.size(0) * 100,
            "loss:",
            running_loss,
        )

    print("__________________________________________________________")
    print("batch_size:", batch_size, "lr:", lr)

    writer.add_hparams(
        {"net": "convnet", "lr": lr, "bsize": batch_size},
        {
            "accuracy": correct / labels.size(0),
            "loss": running_loss,
        },
    )

    torch.save(model, f"convnet_exp_{run_id}_batch_size{batch_size}_lr{lr}.pt")
    model.eval()
    validate(model, test_loader, device, criterion, writer)
