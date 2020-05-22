from data.dataloader.pascal_voc import VOCSegmentation
from torchvision import transforms
# import torch.utils.data as data


# Transforms for Normalization
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
# Create Dataset
trainset = VOCSegmentation(root="F:\\dataset\\voc", split='train', transform=input_transform)
# Create Training Loader

print(len(trainset))
print(trainset[len(trainset) - 1])

