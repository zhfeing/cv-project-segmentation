import os
import argparse
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from utils.visualize import get_color_pallete
from models.deeplabv3 import get_deeplabv3
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Cairo')
def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Demo With Pytorch")
    # model and dataset
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "resnet152"],
                        help="backbone name (default: vgg16)")
    parser.add_argument("--dataset", type=str, default="pascal_voc",
                        choices=["pascal_voc", "pascal_aug", "coco", "citys"],
                        help="dataset name (default: pascal_voc)")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--log-dir", help="Directory for saving checkpoint models")
    parser.add_argument('--state_dict_fp', type=str)
    parser.add_argument(
        "--input-pic",
        type=str,
        help="path to the input picture"
    )
    parser.add_argument(
        "--outdir",
        default="./eval",
        type=str,
        help="path to save the predict result"
    )
    args = parser.parse_args()

    args.model = "deeplabv3"
    return args


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert("RGB")

    img = np.array(image)
    plt.imshow(img)
    plt.show()


    images = transform(image).unsqueeze(0).to(device)

    model = get_deeplabv3(
        dataset=config.dataset,
        backbone=config.backbone,
        pretrained_base=False
    )
    # load parameters
    model.load_state_dict(torch.load(config.state_dict_fp, map_location="cpu"))
    model = model.to(device)
    print("Finished loading model!")

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, config.dataset)
    outname = os.path.splitext(os.path.split(config.input_pic)[-1])[0] + ".png"
    mask.save(os.path.join(config.outdir, outname))


if __name__ == "__main__":
    config = parse_args()
    demo(config)
