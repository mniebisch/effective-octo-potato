"""
Package for the .

Package to provide tools to participate in
Kaggle challenge "Google - Isolated Sign Language Recognition".

author = Michael Nibesch, Christian Reimers
"""

from . import data
from ._models import SimpleNet
from ._split_train_test import split_train_test

__all__ = [
    "data",
    "SimpleNet",
]
