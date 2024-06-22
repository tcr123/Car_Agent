import numpy as np
from PIL import Image 
import torch
from torch import nn
from torchvision import transforms, models
import torchvision.models as models
import io

WORD_DIR = "/home/mustar/car_ws/src/car_assistant/data"

model = models.resnet34()
classes = ['Acura RL Sedan 2012', 'Audi A5 Coupe 2012', 'BMW 3 Series Sedan 2012', 'Cadillac SRX SUV 2012', 'Ford Ranger SuperCab 2011', 'Honda Accord Coupe 2012', 'Hyundai Tucson SUV 2012', 'MINI Cooper Roadster Convertible 2012', 'Toyota Camry Sedan 2012', 'Volvo C30 Hatchback 2012']

def load_model(filepath):
    with open(filepath, 'rb') as f:
        buffer = io.BytesIO(f.read())

    model = torch.load(buffer, map_location=torch.device('cpu'))

    return model

def process_image(image):

    # Process a PIL image for use in a PyTorch model

    # Converting image to PIL image using image file path
    #pil_im = Image.open(image)

    # Building image transform
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    # Transforming image for use with network
    pil_tfd = transform(image)

    # Converting to Numpy array
    array_im_tfd = np.array(pil_tfd)

    return array_im_tfd

def predict(image_path, topk=5):
    # Implement the code to predict the class from an image file

    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_model(f'{WORD_DIR}/model/classifier_10.pth').cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)

    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]

    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]

    print(conf)
    print(predicted)

    return classes[predicted[0]]

