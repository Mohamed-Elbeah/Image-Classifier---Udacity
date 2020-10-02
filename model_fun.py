# Import necessary packages
import torch
from torch import nn
from torchvision import models

from PIL import Image

# Function to build a pretrained model
def build_classifier(kind='resnet18', droupout=0.2):
    
    if kind == 'resnet18':
        ResNet18 = models.resnet18(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in ResNet18.parameters():
            param.requires_grad = False
        # Change classifier part
        ResNet18.fc = nn.Sequential(nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=droupout),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Dropout(p=droupout),
                                    nn.Linear(128, 102),
                                    nn.LogSoftmax(dim=1))
        return ResNet18
   
    else:
        print('Sorry ! {} is not valid'.format(model))

        
# Function to save the model
def save_model(save_dir, train_data, model, optimizer, criterion, epochs, lr):
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'model': 'resnet18',
                  'classifier': model.fc,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optim_state': optimizer.state_dict,
                  'epochs': epochs,
                  'criterion': criterion,
                  'learning_rate': lr}

    return torch.save(checkpoint, save_dir)      
        
# Function to load the model

def load_checkpoint(save_dir):
    checkpoint = torch.load(save_dir)
    ResNet18 = models.resnet18(pretrained=True)
    ResNet18.fc = checkpoint['classifier']
    ResNet18.load_state_dict(checkpoint['state_dict'])
    ResNet18.class_to_idx = checkpoint['class_to_idx']
    
    return ResNet18     