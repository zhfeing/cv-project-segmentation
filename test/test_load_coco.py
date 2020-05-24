from data.dataloader.mscoco import COCOSegmentation
from torchvision import transforms
# import torch.utils.data as data


# Transforms for Normalization
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
])
# Create Dataset
trainset = COCOSegmentation(root="/home/zhfeing/datasets/coco", split='val', transform=input_transform)
# Create Training Loader

print(trainset[0])
print(len(trainset))
print(trainset[len(trainset) - 1])



