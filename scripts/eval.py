import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data.dataloader import get_segmentation_dataset
from models.deeplabv3 import get_deeplabv3
from utils.score import SegmentationMetric
from utils.visualize import get_color_pallete
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation With Pytorch")
    # model and dataset
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "resnet152"],
                        help="backbone name (default: vgg16)")
    parser.add_argument("--dataset", type=str, default="pascal_voc",
                        choices=["pascal_voc", "pascal_aug", "coco", "citys"],
                        help="dataset name (default: pascal_voc)")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--workers", "-j", type=int, default=4,
                        metavar="N", help="dataloader threads")
    # training hyper params
    parser.add_argument("--log-dir", help="Directory for saving checkpoint models")
    # cuda setting
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--state_dict_fp', type=str)
    args = parser.parse_args()

    args.model = "deeplabv3"
    return args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            args.dataset,
            split='val',
            mode='testval',
            transform=input_transform,
            root=args.dataset_root
        )
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.workers,
            pin_memory=True
        )

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_deeplabv3(
            dataset=args.dataset,
            backbone=args.backbone,
            pretrained_base=False,
            norm_layer=BatchNorm2d
        )
        # load parameters
        self.model.load_state_dict(torch.load(args.state_dict_fp, map_location="cpu"))
        self.model = self.model.to(self.device)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}|{:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, len(self.val_loader), pixAcc * 100, mIoU * 100)
            )

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.save_pred = True
    if args.save_pred:
        outdir = 'runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger(
        "semantic_segmentation",
        args.log_dir,
        get_rank(),
        filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
