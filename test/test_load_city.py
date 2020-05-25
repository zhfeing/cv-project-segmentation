from data.dataloader import CitySegmentation

from torchvision import transforms
# import torch.utils.data as data


# Transforms for Normalization
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
])

trainset = CitySegmentation(root="/home/zhfeing/datasets/cityscape", transform=input_transform)
print(trainset[0])
print(len(trainset))
print(trainset[len(trainset) - 1])
