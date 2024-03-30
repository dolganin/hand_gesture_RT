import torchvision.models.mobilenet


def get_mobilenet_model(num_classes: int = 29):
    return torchvision.models.mobilenet.mobilenet_v3_small(num_classes=num_classes)
