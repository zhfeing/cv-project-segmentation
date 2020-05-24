import torch
import os

from utils import distributed


# if __name__ == "__main__":
#     # reference maskrcnn-benchmark
#     num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
#     is_distributed = num_gpus > 1
#     device = "cuda"
#     if is_distributed:
#         torch.cuda.set_device(0)
#         torch.distributed.init_process_group(backend="nccl", init_method="env://")
#         distributed.synchronize()

#     debug_file = torch.load("logs/error_checkpoint.pth")

#     images = debug_file["images"]
#     targets = debug_file["targets"]
#     outputs = debug_file["outputs"]
#     loss_dict = debug_file["loss_dict"]
#     losses = debug_file["losses"]
#     losses_reduced = debug_file["losses_reduced"]
#     model: torch.nn.Module = debug_file["model"].module

#     with torch.no_grad():
#         y = model(images[0:1, ...])
#         print(images.shape)
#         print(torch.isnan(y).any())
#         print(y)

a = torch.load("checkpoints/deeplabv3_resnet50_coco.pth", map_location="cpu")
print(a)