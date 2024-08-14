import os
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms


# Function to parse XML and extract relevant info
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {
        "filename": root.find("filename").text,
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
        "objects": [],
    }

    for obj in root.findall("object"):
        obj_data = {
            "name": obj.find("name").text,
            "pose": obj.find("pose").text,
            "bndbox": {
                "xmin": int(obj.find("bndbox/xmin").text),
                "ymin": int(obj.find("bndbox/ymin").text),
                "xmax": int(obj.find("bndbox/xmax").text),
                "ymax": int(obj.find("bndbox/ymax").text),
            },
        }
        data["objects"].append(obj_data)

    return data


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        bbox = {
            "xmin": int(bndbox.find("xmin").text),
            "ymin": int(bndbox.find("ymin").text),
            "xmax": int(bndbox.find("xmax").text),
            "ymax": int(bndbox.find("ymax").text),
        }
        label = obj.find("name").text
        objects.append({"bbox": bbox, "label": label})

    filename = root.find("filename").text
    return {"filename": filename, "objects": objects}


def inference(model, image_path, device="cpu"):
    # Define the image transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the specified device
    image_tensor = image_tensor.to(device)

    # Set the model to evaluation mode and move to the device
    model.eval()
    model.to(device)

    with torch.no_grad():
        bbox, class_label = model(image_tensor)

    # Convert the bounding box coordinates back to the original image dimensions
    bbox = bbox.squeeze().cpu().numpy()
    bbox[0] *= original_width
    bbox[1] *= original_height
    bbox[2] *= original_width
    bbox[3] *= original_height

    # Get the class label
    class_label = class_label.argmax(dim=1).item()

    return bbox, class_label, image


def plot_image_with_bbox(image, bbox, class_label, class_names):
    plt.imshow(image)
    plt.gca().add_patch(
        plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    plt.text(
        bbox[0],
        bbox[1] - 10,
        class_names[class_label],
        bbox=dict(facecolor="red", alpha=0.5),
        fontsize=12,
        color="white",
    )
    plt.axis("off")
    plt.show()


def get_random_image_path(directory):
    files = os.listdir(directory)
    image_files = [
        file for file in files if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    random_image_file = random.choice(image_files)
    return os.path.join(directory, random_image_file)
