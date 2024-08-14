import copy

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, Precision, Recall
from torchvision import transforms

from dataset import CatDogDataset
from model import SimpleObjectDetectionNet

mlflow.set_tracking_uri("http://127.0.0.1:8080")
cat_dog_experiment = mlflow.set_experiment("CatDogDetector")
run_name = "cat_dog_detector_gdm"
artifact_path = "cat_dog_detector_artifact"

if __name__ == "__main__":
    device = torch.device("cuda")
    model = SimpleObjectDetectionNet(num_classes=2)  # 2 classes: cat and dog
    model.to(device)

    batch_size = 64
    lr = 3e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    classification_criterion = nn.CrossEntropyLoss()
    detection_criterion = nn.MSELoss()

    # Classification metrics
    accuracy = Accuracy(task="multiclass", num_classes=2).to(device)
    precision = Precision(task="multiclass", num_classes=2).to(device)
    recall = Recall(task="multiclass", num_classes=2).to(device)
    f1_score = F1Score(task="multiclass", num_classes=2).to(device)

    # Detection metrics
    mae = MeanAbsoluteError().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = CatDogDataset(
        xml_dir="dataset/annotations/xmls",
        img_dir="dataset/images",
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_epochs = 10
    lowest_loss = 1e10

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "classification_loss_function": classification_criterion.__class__.__name__,
            "detection_loss_function": detection_criterion.__class__.__name__,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            for batch, (images, labels, boxes) in enumerate(dataloader):
                optimizer.zero_grad()
                images, labels, boxes = images.to(device), labels.to(device), boxes.to(device)

                predicted_bbox, predicted_label = model(images)
                _, preds = torch.max(predicted_label, 1)

                classification_loss = classification_criterion(predicted_label, labels)
                detection_loss = detection_criterion(predicted_bbox, boxes)

                loss = classification_loss + detection_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Log metrics
                mlflow.log_metric("loss", loss.item(), step=epoch * len(dataloader) + batch)
                mlflow.log_metric(
                    "classification_loss",
                    classification_loss.item(),
                    step=epoch * len(dataloader) + batch,
                )
                mlflow.log_metric(
                    "detection_loss", detection_loss.item(), step=epoch * len(dataloader) + batch
                )
                mlflow.log_metric(
                    "accuracy", accuracy(preds, labels).item(), step=epoch * len(dataloader) + batch
                )
                mlflow.log_metric(
                    "precision",
                    precision(preds, labels).item(),
                    step=epoch * len(dataloader) + batch,
                )
                mlflow.log_metric(
                    "recall", recall(preds, labels).item(), step=epoch * len(dataloader) + batch
                )
                mlflow.log_metric(
                    "f1_score", f1_score(preds, labels).item(), step=epoch * len(dataloader) + batch
                )
                mlflow.log_metric(
                    "mae", mae(predicted_bbox, boxes).item(), step=epoch * len(dataloader) + batch
                )

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(
                f">>> Epoch [{epoch+1}/{num_epochs}]:    Loss: {epoch_loss:.4f}    Acc: {epoch_acc}"
            )

            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "cat_dog_detector.pth")
        mlflow.pytorch.log_model(model, "cat_dog_detector")
