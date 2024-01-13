import os
import random
import numpy as np
import torch
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# 시드 고정
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
