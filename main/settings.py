# 导入标准库
import math
import random
import paddle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from paddle.nn import Linear, Embedding, Conv2D
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 绘图设置
pd.options.display.notebook_repr_html=True  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题
%matplotlib inline
