# Import necessary packages
import torch
from torch import nn, optim
import argparse
import utility_fun, model_fun


parser = argparse.ArgumentParser(description='Using Neural Networks for Image Classifier')


parser.add_argument('--data_dir', action='store',
                    default = 'C:/Users/mohamedelbeah/home/Image_Classifier/flowers',
                    help='Enter path to data.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'C:/Users/mohamedelbeah/home/Image_Classifier/checkpoint.pth',
                    help='Enter location to save checkpoint')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate', default = 0.001,
                    help='Enter learning rate, default is 0.001')

parser.add_argument('--droupout', action='store',
                    dest='droupout', default = 0.2,
                    help='Enter droupout, default is 0.2')

parser.add_argument('--epochs', action='store',
                    dest='epochs', type=int, default = 25,
                    help='Enter number of epochs, default is 3')


results = parser.parse_args()

data_dir = results.data_dir
save_dir = results.save_directory
lr = results.learning_rate
epochs = results.epochs
droupout = results.droupout


# Loading data
trainloader, validloader, testloader, train_data = utility_fun.load_data(data_dir)

# Building classifier
ResNet18 = model_fun.build_classifier()

# Create criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(ResNet18.fc.parameters(), lr=lr)

# Use GPU if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training the model
ResNet18.to(device)

train_losses, valid_losses = [], []
for e in range(epochs):
    train_loss = 0
    for images, labels in trainloader:
        # Move images and labels tensors to the default device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = ResNet18.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    else:
        # Validation Pass
        valid_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # set model to evaluation mode
            ResNet18.eval()
            
            for images, labels in validloader:
                # Move images and labels tensors to the default device
                images, labels = images.to(device), labels.to(device)
                
                logps = ResNet18.forward(images)
                valid_loss += criterion(logps, labels)
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                result = top_class == labels.view(top_class.shape)
                accuracy += torch.mean(result.type(torch.FloatTensor))
            
        train_losses.append(train_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))
        
        print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                    "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
        
        # set model back to train mode
        ResNet18.train()

# Saving the model
model_fun.save_model(save_dir, train_data, ResNet18, optimizer, criterion, epochs, lr)
