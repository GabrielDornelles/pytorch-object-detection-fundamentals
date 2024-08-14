from utils import inference, plot_image_with_bbox, get_random_image_path
from model import SimpleObjectDetectionNet
import torch

class_names = {0: 'cat', 1: 'dog'}
model = SimpleObjectDetectionNet(num_classes=len(class_names))
model.load_state_dict(torch.load("cat_dog_detector.pth"))
bbox, class_label, image = inference(model, get_random_image_path('dataset/images'), device='cpu')
plot_image_with_bbox(image, bbox, class_label, class_names)