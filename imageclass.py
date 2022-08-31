# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:38:04 2022

@author: cedri
"""

# =============================================================================
# This tutorial follows a basic machine learning workflow:
# 
#     Examine and understand data
#     Build an input pipeline
#     Build the model
#     Train the model
#     Test the model
#     Improve the model and repeat the process
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
