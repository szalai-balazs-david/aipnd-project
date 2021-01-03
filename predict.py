import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
import json
import sys
import argparse
import pathlib


# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['basemodel'] == 'vgg16':
        loaded_model = models.vgg16(pretrained=True)
    elif checkpoint['basemodel'] == 'vgg13':
        loaded_model = models.vgg13(pretrained=True)
    elif checkpoint['basemodel'] == 'vgg11':
        loaded_model = models.vgg11(pretrained=True)
    else:
        raise Exception('Unknown basemodel type')
    
    classifier = nn.Sequential(checkpoint['classifier_definition'])
    
    classifier.load_state_dict(checkpoint['classifier'])
    
    loaded_model.classifier = classifier
    
    loaded_model.class_to_idx = checkpoint['class_to_index']
    
    loaded_model.eval()
    
    return loaded_model
    
    
def process_image(image):
    #Using same transforms as in original test setup
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    img = data_transforms(image)
    return img


def predict(image_path, model, map_file, topk=5, gpu = True, verbose = True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    print(f'Device used: {device}', flush=True)
    
    #Open image and execute transformations on it
    img = Image.open(image_path)
    processed_img = process_image(img)
    processed_img = processed_img.unsqueeze(dim = 0)
    processed_img = processed_img.to(device)
    
    #Feed processed image through the loaded network
    with torch.no_grad():
        loaded_model = load_checkpoint(model)
        loaded_model.to(device)
        log_ps = loaded_model.forward(processed_img)

        ps = torch.exp(log_ps)
        
        probs, classes = ps.topk(topk)
        
        probs = probs.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]
        
        # Match classes to flower names
        idx_to_class = {value: key for key, value in loaded_model.class_to_idx.items()}    
        indices = [idx_to_class[i] for i in classes]
        
    if map_file == '':
        return probs, indices
    else:
        with open(map_file, 'r') as f:
            cat_to_name = json.load(f)
        flowers = [cat_to_name[i] for i in indices]
        
        return probs, flowers
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network on a set of images.')
    parser.add_argument('img', type=pathlib.Path,
                        help='Path to the target image.')
    parser.add_argument('checkpoint', type=pathlib.Path,
                        help='Path to neural network checkpoint file')
    parser.add_argument('--category_names', type=str, default='',
                        help='Path to class name mapping file.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of best matches to return.')
    parser.add_argument('--gpu', action='store_true',
                        help='Should use GPU / CUDA? (If available.)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose')

    args = parser.parse_args()
    
    probs, items = predict(image_path = str(args.img), model = str(args.checkpoint), map_file = args.category_names, topk = args.top_k, gpu = args.gpu, verbose = args.verbose)
    for i, (prob) in enumerate(probs):
        print(f'{items[i]}: {probs[i] * 100:.1f}%')