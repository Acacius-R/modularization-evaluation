import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, num_classes=10):
    if model_name == 'rescnn':
        from models.rescnn import ResCNN
        model = ResCNN(num_classes=num_classes)
    elif model_name == 'vgg16':
        from models.vgg16 import VGG16
        model = VGG16(num_classes=num_classes)
    elif model_name == 'simcnn':
        from models.simcnn import SimCNN
        model = SimCNN()
    else:
        raise ValueError
    return model


def load_trained_model(model_name, n_classes, trained_model_path):
    model = load_model(model_name, num_classes=n_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model