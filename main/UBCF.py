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

# 定义数据读取函数
def load_data(file_dir):
    user_id_dict, rated_mov_id_dict = {},{}
    u_idx, m_idx, r_idx = 0, 0, 0
    data_list = []
    
    f = open(file_dir)
    for line in f.readlines():
        usr,mov,rating,_ = line.split('::')
        if int(usr) not in user_id_dict:
            user_id_dict[int(usr)] = u_idx
            u_idx += 1
        if int(mov) not in rated_mov_id_dict:
            rated_mov_id_dict[int(mov)] = m_idx
            m_idx += 1
        # 标准化用户ID和电影ID
        data_list.append([r_idx,user_id_dict[int(usr)],rated_mov_id_dict[int(mov)],float(rating)])
        r_idx += 1
    f.close()
    
    N,M = u_idx,m_idx
    
    return data_list,user_id_dict,rated_mov_id_dict,N,M

# 读取评分数据
rating_dir = "./ratings.dat"
ratings_list, user_id_dict, movie_id_dict, N, M = load_data(rating_dir)
print('Total User Number: %d \nTotal Movie Number: %d'%(N,M))
# 电影ID对应字典
movie_id_dict = {value:key for key,value in movie_id_dict.items()}
# 用户ID对应字典
user_id_dict = {value:key for key,value in user_id_dict.items()}

# 数据集划分
# 按照8:2随机划分ratings数据集
train_list, test_list = train_test_split(ratings_list, test_size=0.2, random_state=10)
# 训练集（含索引）
train_idx = [x[0] for x in train_list]
train_ratings_list = [x[1:] for x in train_list]
# 测试集（含索引）
test_idx = [x[0] for x in test_list]
test_ratings_list = [x[1:] for x in test_list]
print('Train Size: %d \nTest Size: %d'%(len(train_ratings_list),len(test_ratings_list)))

# 定义数据格式转换函数
def list2matrix(sequence, N, M):
    """定义格式转换函数：
    1. 将list转换成matrix二维数组
    2. 输出用户-电影评分矩阵
    """
    array_records = np.array(sequence)
    mat = np.zeros([N,M])
    row = array_records[:,0].astype(int)
    col = array_records[:,1].astype(int)
    value = array_records[:,2].astype(float)
    mat[row,col] = value
    return mat

# 得到评分矩阵
train_ratings_mat = list2matrix(train_ratings_list, N, M)
test_ratings_mat = list2matrix(test_ratings_list, N, M)
# 训练集
print("User-Item Matrix shape:",train_ratings_mat.shape)
spar = np.sum(train_ratings_mat==0)/train_ratings_mat.size
print("User-Item Matrix sparsity: %.4f"%spar)
pd.DataFrame(train_ratings_mat)
# 测试集
print("User-Item Matrix shape:",test_ratings_mat.shape)
test_ratings_df = pd.DataFrame(test_ratings_mat)
spar = np.sum(test_ratings_mat==0)/test_ratings_mat.size
print("User-Item Matrix sparsity: %.4f"%spar)
test_ratings_df

# 保存评分矩阵
train_ratings_df = pd.DataFrame(train_ratings_mat)
train_ratings_df.to_csv("rating_matrix.csv", index=0)
