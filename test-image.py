import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CSRNet
import numpy as np
import cv2
from matplotlib import cm as c
from matplotlib import pyplot as plt


# Load the model
model = CSRNet()
model = model.cuda()
checkpoint = torch.load('partBmodel_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Preprocess the input image
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # Normalize pixel values
        std=[0.229, 0.224, 0.225]
    )
])
image_path = 'ext_testing/images/348579209_798317721511766_867196015875938002_n.jpg'
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).cuda()


# Run the input through the model
with torch.no_grad():
    output = model(input_image)
count = output.cpu().sum().item()


# Plot the output
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

countText = f"Predicted Count : {int(count)}"
axes[0].text(0, -60, countText, fontsize=16)

heatmap = np.asarray(output.cpu().reshape(
    output.cpu().shape[2], output.cpu().shape[3]))
axes[0].imshow(heatmap, cmap=c.jet)
axes[0].set_title("Heatmap")

axes[1].imshow(plt.imread(image_path))
axes[1].set_title("Original Image")

alpha = 0.5
resized_heatmap = cv2.resize(
    heatmap, (input_image.shape[3], input_image.shape[2]))
axes[2].imshow(plt.imread(image_path), alpha=1-alpha)
axes[2].imshow(resized_heatmap, cmap=c.jet, alpha=alpha)
axes[2].set_title("Overlay")

plt.subplots_adjust(top=0.7, bottom=0.15, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
plt.show()
