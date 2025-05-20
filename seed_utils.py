import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto, Session

def setar_semente(valor=42):
    os.environ['PYTHONHASHSEED'] = str(valor)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(valor)
    np.random.seed(valor)
    tf.random.set_seed(valor)
    print(f"Semente definida: {valor}")