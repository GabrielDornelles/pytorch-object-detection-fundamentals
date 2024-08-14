import os
import torch
from PIL import Image
from utils import parse_voc_annotation


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, xml_dir, img_dir, transform=None, class_to_idx=None):
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = [
            parse_voc_annotation(os.path.join(xml_dir, xml_file))
            for xml_file in os.listdir(xml_dir)
        ]
        self.class_to_idx = class_to_idx if class_to_idx else {"cat": 0, "dog": 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation["filename"])
        image = Image.open(img_path).convert("RGB")

        original_width, original_height = image.size
        boxes = []
        labels = []
        # print(annotation)
        for obj in annotation["objects"]:
            bndbox = obj["bbox"]
            boxes.append([bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]])
            #

            labels.append(self.class_to_idx[obj["label"]])

        boxes = torch.tensor(boxes[0], dtype=torch.float32)
        labels = torch.tensor(labels[0], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            new_width, new_height = image.size(2), image.size(1)

            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Re-scale
            boxes[0] *= scale_x
            boxes[1] *= scale_y
            boxes[2] *= scale_x
            boxes[3] *= scale_y

            # Normalize in 0-1 range
            boxes[0] /= new_width
            boxes[1] /= new_height
            boxes[2] /= new_width
            boxes[3] /= new_height

        return image, labels, boxes
