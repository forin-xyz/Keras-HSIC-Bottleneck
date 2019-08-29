#coding=utf8
"""

# Author : rui
# Created Time : Aug 28 21:59:38 2019
# Description:
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import mnist

def load_mnist_datasets():
    return (
        mnist.train_images(),
        mnist.train_labels(),
        mnist.test_images(),
        mnist.test_labels())


del division
del print_function
del absolute_import
del unicode_literals
