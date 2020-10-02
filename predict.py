# Import necessary packages
import torch
import argparse
import json
import utility_fun, model_fun
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='Using Neural Networks for Image Classifier')

parser.add_argument('--image_path', action='store',
                    default = 'C:/Users/mohamedelbeah/home/Image_Classifier/flowers/test/76/image_02550.jpg',
                    help='Enter path to image')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'C:/Users/mohamedelbeah/home/Image_Classifier/checkpoint.pth',
                    help='Enter location to save checkpoint')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 5,
                    help='Enter number of top most likely classes to view, default is 5')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'C:/Users/mohamedelbeah/home/Image_Classifier/cat_to_name.json',
                    help='Enter path to image.')


results = parser.parse_args()

save_dir = results.save_directory
image = results.image_path
top_k = results.topk
cat_names = results.cat_name_dir

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Loading the model
ResNet18 = model_fun.load_checkpoint(save_dir)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Performing Prediction
ResNet18.to(device)
image_tensor = utility_fun.process_image(image)
image_tensor = image_tensor.unsqueeze_(0)
    
# Turn off gradients
with torch.no_grad():
    # set model to evaluation mode
    ResNet18.eval()
    logps = ResNet18.forward(image_tensor.to(device))
    
# Top k probabilities and classes
ps = torch.exp(logps)
probs, classes = ps.topk(top_k, dim=1)

# Convert probs and classes to arrays
probs = probs.cpu().data.numpy().squeeze()
classes = classes.cpu().data.numpy().squeeze()

# Converting topk indices into actual flower names
idx_to_class = {value: key for key, value in ResNet18.class_to_idx.items()}
labels = [idx_to_class[key] for key in classes]
flower_names = [cat_to_name[key].title() for key in labels]

# TODO: Display an image along with the top 5 classes
fig, (ax1, ax2) = plt.subplots(figsize=(10, 8), ncols=2)
utility_fun.imshow(utility_fun.process_image(image), ax=ax1)
ax1.set_title(flower_names[0])
ax1.axis('off')

ax2.barh(np.arange(5), probs)
ax2.set_xlim(0, 1)
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(5))
ax2.set_yticklabels(flower_names)
ax2.set_title('Class Probability')
ax2.invert_yaxis()                   # labels read top-to-bottom

plt.tight_layout()
plt.show()
