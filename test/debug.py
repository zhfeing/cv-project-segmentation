from data.dataloader import COCOSegmentation
from torchvision import transforms
import torch
import tqdm


# input_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
# ])
# dataset = COCOSegmentation(root="/home/zhfeing/datasets/coco", transform=input_transform)

# for i in tqdm.trange(len(dataset)):
#     x: torch.Tensor = dataset[i][0]
#     y: torch.Tensor = dataset[i][1]
#     # print(x.mean(), x.var(), x.max(), x.min(), sep="\t")
#     # print(y.max(), y.min())
#     # if y.max() == y.min():
#         # print("yyy", y.max(), y.min(), sep="\t")
#     if torch.abs(x.max() - x.min()) < 1e-3:
#         raise Exception("gggg")

#     # print()

pth = torch.load("logs/error_15.pth", map_location="cpu")
images = pth["images"]
targets = pth["targets"]

for i in range(images.shape[0]):
    print(images[i].min(), images[i].max(), images[i].mean(), images[i].var(), sep="\t")
    print(targets[i].min(), targets[i].max(), sep="\t")
