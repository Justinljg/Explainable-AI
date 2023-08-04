import torch
from torchvision import transforms
from PIL import Image as PilImage
from ..train import ImageClassificationModel


def read_imagefile(image):
    img = PilImage.open(image).convert('RGB')

    # set up transforms for your images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # resize the image to 224x224
    img = img.resize((224, 224))

    # apply the specified transforms
    img = transform(img)
    return img

def predict(img) -> str:
    
    model = ImageClassificationModel(num_classes=3)
    model.load_state_dict(torch.load('model_assets/model.pt'))
    model.eval() 
    print("Model loaded")
    output = model(img.unsqueeze(0))
    _, predicted_class = torch.max(output, 1)

    if predicted_class.item() == 1:
        return "This patient has a Pneumonia caused by Bacteria."
    elif predicted_class.item() == 2:
        return "This patient has a Pneumonia caused by Virus."
    elif predicted_class.item() == 0:
        return "This patient is Normal."
    else:
        return "please check labels"

