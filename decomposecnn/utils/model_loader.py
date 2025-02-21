from models.resnet import resnet20
def load_model(model_name, num_classes):
    if model_name == 'resnet20':
        return resnet20()
    else:
        raise ValueError(f"Unknown model name: {model_name}")