"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .pascal_voc import VOCSegmentation

datasets = {
    'pascal_voc': VOCSegmentation,
    'citys': CitySegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
