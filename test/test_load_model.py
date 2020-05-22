from models.deeplabv3 import get_deeplabv3


model = get_deeplabv3(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained_base=True,
    root="F:\\pretrained-models\\checkpoints"
)

print(model)
