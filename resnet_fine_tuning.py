import torchvision.models as models
from torchvision import datasets, models, transforms
from zstd import ZSTD_compress
import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from spectral import *

data_dir = "imagenet/"
input_size = 224
num_epochs = 100
batch_size = 1


def get_model():
    model = models.resnet18(pretrained=True)
    model.cuda()

    q_net = spectral(model)

    return model, q_net


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
# Create training and validation dataloaders
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
    )
    for x in ["train", "val"]
}

model, q_net = get_model()

'''optimizer = torch.optim.Adam(
    list(model.parameters()) + list(q_net.parameters()), lr=1e-3
)'''
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(q_net.parameters()), lr=1e-3, momentum=0.0
)

for epoch in range(num_epochs):
    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for index, (inputs, labels) in enumerate(dataloaders[phase]):
            step = epoch * len(dataloaders[phase]) + index
            num_steps = len(dataloaders[phase]) * num_epochs
            target_bits = (float(num_steps - step) / float(num_steps)) * 6.0 + 3.5

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)

                reconstruction_loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                q_loss = quantization_loss(model)

                '''if q_loss.item() < target_bits:
                    for parameter in q_net.parameters():
                        parameter.requires_grad_(False)
                else:
                    for parameter in q_net.parameters():
                        parameter.requires_grad_(True)'''

                #loss = reconstruction_loss
                loss = reconstruction_loss + q_loss

                print("q_loss", q_loss)
                print("target_bits", target_bits)
                print("reconstruction_loss", reconstruction_loss)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

    print()
