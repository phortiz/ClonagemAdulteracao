import numpy as np
from PIL import Image
from torchvision import  transforms, models
import torch.nn as nn
import torch

# Applying Transforms to the Data
image_transforms = { 
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])}

transform = image_transforms['test']



   
def carregaTensor_to_numpy(image_Path,model):
    


    #retira o último estágio
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)

    transform = image_transforms['test']

    test_image = Image.open(image_Path).convert('RGB')

    test_image_tensor = transform(test_image)

    input_batch = test_image_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    #return output.detach().numpy()
    return output.detach().numpy()



