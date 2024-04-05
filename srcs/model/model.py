import torchvision


def get_mobilenet_model(num_classes: int = 29):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.efficientnet_b0)
    model.classifier[-1].out_features = num_classes
    return model

