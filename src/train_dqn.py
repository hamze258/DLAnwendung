import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from environment import FlappyBirdEnv  # Deine Umgebungsklasse
