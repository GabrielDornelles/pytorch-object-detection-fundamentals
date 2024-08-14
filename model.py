import torch.nn as nn
import torchvision.models as models


class SimpleObjectDetectionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Load the pretrained ResNet-18 model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the fully connected layer of ResNet-18
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Flatten the output from the backbone
        self.flatten = nn.Flatten()

        # Bounding box regression head
        self.bboxHead = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        # Classification head
        self.classificationHead = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)

        bbox = self.bboxHead(x)
        class_label = self.classificationHead(x)

        return bbox, class_label
