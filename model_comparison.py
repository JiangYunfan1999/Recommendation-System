#####################################
############# 数据稀疏性 #############
#####################################

### 计算用户相似度矩阵
################# UBCF 修正余弦相似度 #################
# 计算两个矩阵行向量之间的修正余弦相似度
def cal_rating_cosine(a,b,min_common_num=10):
    assert a.shape == b.shape
    dim = len(a.shape)
    # 判断是否共同评分
    common_movies = a*b > 0
    # 共同评分电影数量
    common_num = np.sum(common_movies, axis=dim-1)
    
    mean_a = np.sum(a, axis=dim-1)/np.sum(a>0, axis=dim-1)
    mean_b = np.sum(b, axis=dim-1)/np.sum(b>0, axis=dim-1)
    if dim == 1:  # 若a为列向量
        new_a = (a - mean_a)*common_movies
        new_b = (b - mean_b)*common_movies
    else:
        new_a = (a - mean_a.reshape((-1,1)))*common_movies
        new_b = (b - mean_b.reshape((-1,1)))*common_movies
        
    sim = np.sum(new_a*new_b, axis=dim-1)/(np.sqrt(np.sum(new_a**2, axis=dim-1))*np.sqrt(np.sum(new_b**2, axis=dim-1)) + 1e-10)
    
    return sim*(common_num > min_common_num)

# 得到相似度矩阵
def get_similarity_matrix(mat):
    n, m = mat.shape
    sim_list = []
    for u in range(n):
        a = np.tile(mat[u,:], (n,1))  # 平铺成二维矩阵
        b = mat
        # 该用户与其他用户的相似度
        sim = cal_rating_cosine(a, b)
        sim_list.append(sim)
        if u%1000 == 0:
            print("have calculated %d users..."% u)
    print("End.")
    sim_mat = np.array(sim_list)
    return sim_mat

# 修正余弦相似度（共同评分数=10）
sim_knn = get_similarity_matrix(mat = train_ratings_mat)

################# MFBCF 皮尔森相关系数 #################
# 皮尔森相关系数
sim_mf = np.corrcoef(P)

################# DLBCF 皮尔森相关系数 #################
# 皮尔森相关系数
sim_dl = np.corrcoef(user_features)

# 保存相似度矩阵
pd.DataFrame(sim_knn).to_csv("similarity_UBCF.csv", index=0)
pd.DataFrame(sim_mf).to_csv("similarity_MFBCF.csv", index=0)
pd.DataFrame(sim_dl).to_csv("similarity_DLBCF.csv", index=0)

### 相似度分布
# 取消科学计数法输出
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x:'%.4f'%x)

# 忽略对角线上sim=1
mask = np.array(np.eye(6040)-1, dtype = bool)

# 描述统计量
print("KNN:")
print(pd.DataFrame(sim_knn[mask].reshape(-1)).describe())
print("MF:")
print(pd.DataFrame(sim_mf[mask].reshape(-1)).describe())
print("DL:")
print(pd.DataFrame(sim_dl[mask].reshape(-1)).describe())

df = pd.DataFrame(sim_knn[mask].reshape(-1),columns=['UBCF'])
df['MFBCF'] = sim_mf[mask].reshape(-1)
df['DLBCF'] = sim_dl[mask].reshape(-1)

plt.title('Distribution of similarity')
sns.boxplot(data = df)
plt.show()

### 相似度的差异性
random.seed(2022)
idx = random.sample(range(len(sim_mf)),10)
定义绘制热力图函数
def plt_similarity_matrix(mat,idx):
    # 随机选择用户
    plt_mat = mat[idx][:,idx]
    #生成一个全为0的矩阵 
    mask = np.zeros_like(plt_mat)
    #将mask的对角线及以上设置为True，对应要被遮掉的部分
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=(8, 8))
    sns.heatmap(plt_mat, mask=mask, annot=True, vmax=1, square=True, cmap="RdBu_r")
    plt.show()

# KNN
plt_similarity_matrix(sim_knn,idx)
# MF
plt_similarity_matrix(sim_mf,idx)
# DL
plt_similarity_matrix(sim_dl,idx)

### K近邻的差异性
# 读入保存的相似度矩阵
sim_knn = pd.read_csv('similarity_UBCF.csv')
sim_mf = pd.read_csv('similarity_MFBCF.csv')
sim_dl = pd.read_csv('similarity_DLBCF.csv')
sim_knn = np.array(sim_knn)
sim_mf = np.array(sim_mf)
sim_dl = np.array(sim_dl)

# 计算近邻重合度
def k_neighbor_compare(sim_a, sim_b, k_num):
    assert sim_a.shape==sim_b.shape
    overlap_num = []
    # 随机选取50位用户，计算平均重合度
    random.seed(123)
    usr_idx = random.sample(range(len(sim_a)),50)
    for i in usr_idx:
        neighbor_set_a = np.argsort(-sim_a)[i, :k_num]
        neighbor_set_b = np.argsort(-sim_b)[i, :k_num]
        num = len(set(neighbor_set_a) & set(neighbor_set_b))
        print(num,end=" ")
        overlap_num.append(num)
    print()
    return np.mean(overlap_num)

k_num = 30  # k近邻

ratio1 = k_neighbor_compare(sim_knn, sim_mf, k_num=k_num)/k_num
ratio2 = k_neighbor_compare(sim_mf, sim_dl, k_num=k_num)/k_num
ratio3 = k_neighbor_compare(sim_knn, sim_dl, k_num=k_num)/k_num

print("Common Neighbor Ratio:")
print("KNN vs MF:",ratio1)
print("MF vs DL:",ratio2)
print("DL vs KNN:",ratio3)

#####################################
############# 预测精度 ##############
#####################################

################# UBCF 相似度加权评分 #################
# 定义相似度加权评分
def weighted_predict(train_mat, sim_mat, K=1):
    assert len(train_mat.shape)>1
    train_mat = train_mat.astype(np.float32)
    
    n,m = train_mat.shape
    
    sim_sort = -1*np.sort(-np.array(sim_mat))[:,1:K+1]   # 除去用户自己
    neighbors = np.argsort(-np.array(sim_mat))[:,1:K+1]  # 近邻的idx
    
    common_items = train_mat[neighbors] 
    mean_user = np.reshape(np.sum(train_mat,axis=1)/np.sum(train_mat>0,axis=1), (-1,1))
    mat_m = train_mat - mean_user
    
    aa = np.sum(sim_sort[:,:,np.newaxis]*mat_m[neighbors]*common_items,axis=1)
    bb = np.sum(sim_sort[:,:,np.newaxis]*common_items,axis=1)+1e-10 # 1e-10保证分母不为０
    r_pred = mean_user + aa/bb
        
    return r_pred

K = 5  # 设置为5~30
pred_knn = weighted_predict(train_mat=train_ratings_mat, sim_mat=sim_knn, K=K)
mae, rmse, acc = mae_rmse_acc(r_pred=pred_knn, test_mat=test_ratings_mat)
print('K=',K)
print('MAE:%.4f, RMSE:%.4f, Accuracy: %.4f'%(mae,rmse,acc))

# 绘图：评价指标随K值变化
k_list = [[0.8479, 1.0768, 0.0040],[0.8488, 1.0829, 0.0042],[0.8466, 1.0827, 0.0044],
         [0.8439, 1.0806, 0.0045],[0.8408, 1.0780, 0.0046],[0.8368, 1.0737, 0.0046]]
k_df = pd.DataFrame(k_list,columns=['MAE','RMSE','ACC'],index=[5,10,15,20,25,30])

plt.title("MAE of UBCF in different K")
plt.xticks(np.arange(5,35,5))
plt.xlabel('Number of Neighbors: K')
fig = sns.lineplot(data=k_df[['MAE']], markers=True)
fig.set_ylim(0.83,0.86)
plt.show()

plt.title("RMSE of UBCF in different K")
plt.xticks(np.arange(5,35,5))
plt.xlabel('Number of Neighbors: K')
fig = sns.lineplot(data=k_df[['RMSE']], markers=True)
fig.set_ylim(1.06,1.09)
plt.show()

plt.title("Accuracy of UBCF in different K")
plt.xticks(np.arange(5,35,5))
plt.xlabel('Number of Neighbors: K')
fig = sns.lineplot(data=k_df[['ACC']], markers=True)
fig.set_ylim(0.002,0.007)
plt.show()

################# MFBCF 矩阵乘积评分 #################
# 读入隐因子矩阵
P = pd.read_csv("user_latent_factors_10.csv")
Q = pd.read_csv("movie_latent_factors_10.csv")
pred_mf = np.dot(np.array(P),np.array(Q).T)
# 输入K=10维
pred_mf = mf_model.prediction(P, Q)
# 模型评估
mae, rmse, acc = mae_rmse_acc(r_pred=pred_mf, test_mat=test_ratings_mat)
print('MF rating prediction:')
print('MAE:%.4f, RMSE:%.4f, Accuracy: %.4f'%(mae,rmse,acc))

################# DLBCF 特征向量相似性评分 #################
# 计算两个矩阵对应行向量之间的余弦相似度
def cal_row_vec_cosine(a,b):
    assert a.shape == b.shape
    dim = len(a.shape)
    
    mean_a = np.sum(a, axis=dim-1)/np.sum(a>0, axis=dim-1)
    mean_b = np.sum(b, axis=dim-1)/np.sum(b>0, axis=dim-1)
    if dim == 1:  # 若a为列向量
        new_a = (a - mean_a)
        new_b = (b - mean_b)
    else:
        new_a = (a - mean_a.reshape((-1,1)))
        new_b = (b - mean_b.reshape((-1,1)))
        
    sim = np.sum(new_a*new_b, axis=dim-1)/(np.sqrt(np.sum(new_a**2, axis=dim-1))*np.sqrt(np.sum(new_b**2, axis=dim-1)) + 1e-10)
    return sim

# 得到相似度矩阵
def feat_sim_predict(usr_mat, mov_mat):
    n = len(usr_mat)
    m = len(mov_mat)
    r_pred = []
    for u in range(n):
        a = np.tile(usr_mat[u,:], (m,1))  # 平铺成二维矩阵
        b = mov_mat
        # 该用户特征与所有电影特征的相似度
        pred = cal_row_vec_cosine(a, b)
        # 标准化scale to 0~5
        r_pred.append(pred*5)
        
    r_pred = np.array(r_pred)
    return r_pred
    
# 读取保存的特征
user_features = pd.read_csv('dl_user_features.csv')
movie_features = pd.read_csv('dl_movie_features.csv')

# 输入32维
pred_dl = feat_sim_predict(np.array(user_features),np.array(movie_features))

# 模型评估
mae, rmse, acc = mae_rmse_acc(r_pred=pred_dl, test_mat=test_ratings_mat)
print('DL rating prediction:')
print('MAE:%.4f, RMSE:%.4f, Accuracy: %.4f'%(mae,rmse,acc))

# 使用模型训练过程中的模型评估
epoch = 1

MAE, RMSE, ACC = evaluation(model, "./checkpoint/epoch1.pdparams")
print("Epoch "+str(epoch))
print("  MAE: %.4f, RMSE: %.4f, ACC: %.4f"%(MAE, RMSE, ACC))

#####################################
############# 推荐质量 ##############
#####################################

# 得到TopN推荐列表
def get_topn(r_pred, train_mat, n=10):
    unrated_items = r_pred * (train_mat==0)
    idx = np.argsort(-unrated_items)
    return idx[:,:n]

# 召回率、精确率
def recall_precision(topn, test_mat, score_level=0):
    n,m = test_mat.shape
    hits,total_pred,total_true = 0.,0.,0.
    # 随机选择1000个用户
    random.seed(1234)
    usr_list = random.sample(range(0,6040),1000)
    for u in usr_list:
        # 有评分且推荐的数量
        hits += len([i for i in topn[u,:] if test_mat[u,i]>score_level])
        # 推荐的电影数量
        size_pred = len(topn[u,:])
        # 有评分的数量
        size_true = np.sum(test_mat[u,:]>score_level,axis=0)
        total_pred += size_pred
        total_true += size_true

    recall = hits/total_true
    precision = hits/total_pred
    return recall, precision

################# UBCF #################
n = 20
# 推荐列表
topn_knn = get_topn(r_pred=pred_knn, train_mat=train_ratings_mat, n=n)
# 推荐效果
recall, precision = recall_precision(topn=topn_knn, 
                                     test_mat=train_ratings_mat,
                                     score_level=3)
print("KNN:")
print("Recall:%.4f, Precision:%.4f"%(recall,precision))

################# MFBCF #################
n = 20
# 推荐列表
topn_mf = get_topn(r_pred=pred_mf, train_mat=train_ratings_mat, n=n)
# 推荐效果
recall, precision = recall_precision(topn=topn_mf, 
                                     test_mat=train_ratings_mat,
                                     score_level=3)
print("MF:")
print("Recall:%.4f, Precision:%.4f"%(recall,precision))

################# DLBCF #################
n = 20
# 推荐列表
topn_dl = get_topn(r_pred=pred_dl, train_mat=train_ratings_mat, n=n)
# 推荐效果
recall, precision = recall_precision(topn=topn_dl, 
                                     test_mat=train_ratings_mat,
                                     score_level=3)
print("DL:")
print("Recall:%.4f, Precision:%.4f"%(recall,precision))

# 推荐结果重合度
def topn_compare(topn_a, topn_b):
    assert topn_a.shape==topn_b.shape
    overlap_num = []
    # 随机选取1000位用户，计算平均重合度
    usr_idx = random.sample(range(len(topn_a)),1000)
    for i in usr_idx:
        topn_set_a = topn_a[i,:]
        topn_set_b = topn_b[i,:]
        num = len(set(topn_set_a) & set(topn_set_b))
        overlap_num.append(num)
    return np.mean(overlap_num)

n_num = 20

ratio1 = topn_compare(topn_knn, topn_mf)/n_num
ratio2 = topn_compare(topn_mf, topn_dl)/n_num
ratio3 = topn_compare(topn_dl, topn_knn)/n_num

print("Common Recommendation Ratio:")
print("KNN vs MF:",ratio1)
print("MF vs DL:",ratio2)
print("DL vs KNN:",ratio3)

# 推荐内容与用户契合度
def recommend_mov(usr_id, topn, mov_info_path, movie_id_dict):
    mov_info = {}
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item
    print("User:", usr_id)
    print("Recommend List:")
    
    usr_topn = topn[usr_id,:]
    for idx in usr_topn:
        id = movie_id_dict[idx]
        print("Movie:", id, mov_info[str(id)])

movie_data_path = "movies.dat"
usr_id = 1957

print("KNN:")
recommend_mov(usr_id, topn_knn, movie_data_path, movie_id_dict)
print("MF:")
recommend_mov(usr_id, topn_mf, movie_data_path, movie_id_dict)
print("DL:")
recommend_mov(usr_id, topn_dl, movie_data_path, movie_id_dict)

########## 深度学习的推荐方法
###################### 保存特征

from PIL import Image
# 加载第三方库Pickle，用来保存Python数据到本地
import pickle
# 定义特征保存函数
def get_usr_mov_features(model, params_file_path, poster_path):
    paddle.set_device('cpu') 
    usr_pkl = {}
    mov_pkl = {}
    
    # 定义将list中每个元素转成tensor的函数
    def list2tensor(inputs, shape):
        inputs = np.reshape(np.array(inputs).astype(np.int64), shape)
        return paddle.to_tensor(inputs)

    # 加载模型参数到模型中，设置为验证模式eval（）
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()
    # 获得整个数据集的数据
    dataset = model.Dataset.dataset

    for i in range(len(dataset)):
        # 获得用户数据，电影数据，评分数据  
        # 本案例只转换所有在样本中出现过的user和movie，实际中可以使用业务系统中的全量数据
        usr_info, mov_info, score = dataset[i]['user_info'], dataset[i]['movie_info'],dataset[i]['scores']
        usrid = str(usr_info['user_id'])
        movid = str(mov_info['movie_id'])

        # 获得用户数据，计算得到用户特征，保存在usr_pkl字典中
        if usrid not in usr_pkl.keys():
            usr_id_v = list2tensor(usr_info['user_id'], [1])
            usr_age_v = list2tensor(usr_info['age'], [1])
            usr_gender_v = list2tensor(usr_info['gender'], [1])
            usr_job_v = list2tensor(usr_info['job'], [1])

            usr_in = [usr_id_v, usr_gender_v, usr_age_v, usr_job_v]
            usr_feat = model.get_usr_feat(usr_in)

            usr_pkl[usrid] = usr_feat.numpy()
        
        # 获得电影数据，计算得到电影特征，保存在mov_pkl字典中
        if movid not in mov_pkl.keys():
            mov_id_v = list2tensor(mov_info['movie_id'], [1])
            mov_tit_v = list2tensor(mov_info['title'], [1, 1, 15])
            mov_cat_v = list2tensor(mov_info['category'], [1, 6])

            mov_in = [mov_id_v, mov_cat_v, mov_tit_v]
            mov_feat = model.get_mov_feat(mov_in)

            mov_pkl[movid] = mov_feat.numpy()
    
    print(len(usr_pkl.keys()))
    print(len(mov_pkl.keys()))
    # 保存特征到本地
    pickle.dump(usr_pkl, open('./usr_feat.pkl', 'wb'))
    pickle.dump(mov_pkl, open('./mov_feat.pkl', 'wb'))
    print("usr / mov features saved!!!")


param_path = "./checkpoint/epoch1.pdparams"
poster_path = "./"
get_usr_mov_features(model, param_path, poster_path) 

###################### 读取特征
mov_feat_dir = 'mov_feat.pkl'
usr_feat_dir = 'usr_feat.pkl'

usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
mov_feats = pickle.load(open(mov_feat_dir, 'rb'))

# 电影特征的路径
movie_data_path = "./movies.dat"
mov_info = {}
# 打开电影数据文件，根据电影ID索引到电影信息
with open(movie_data_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item

usr_file = "./users.dat"
usr_info = {}
# 打开文件，读取所有行到data中
with open(usr_file, 'r') as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        usr_info[str(item[0])] = item

###################### 推荐电影

# 计算目标用户和所有电影的相似度，构建相似度矩阵
import paddle

# 根据用户ID获得该用户的特征
usr_ID = 2022
# 读取保存的用户特征
usr_feat_dir = 'usr_feat.pkl'
usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
# 根据用户ID索引到该用户的特征
usr_ID_feat = usr_feats[str(usr_ID)]

# 记录计算的相似度
cos_sims = []
# 记录下与用户特征计算相似的电影顺序

# with dygraph.guard():
paddle.disable_static()
# 索引电影特征，计算和输入用户ID的特征的相似度
for idx, key in enumerate(mov_feats.keys()):
    mov_feat = mov_feats[key]
    usr_feat = paddle.to_tensor(usr_ID_feat)
    mov_feat = paddle.to_tensor(mov_feat)
    
    # 计算余弦相似度
    sim = paddle.nn.functional.common.cosine_similarity(usr_feat, mov_feat)
    # 打印特征和相似度的形状
    # 从形状为（1，1）的相似度sim中获得相似度值sim.numpy()[0]，并添加到相似度列表cos_sims中
    cos_sims.append(sim.numpy()[0])
    

# 对相似度排序，获得最大相似度在cos_sims中的位置
index = np.argsort(cos_sims)
# 打印相似度最大的前topk个位置
topk = 20
print("相似度最大的前{}个索引是{}\n对应的相似度是：{}\n".format(topk, index[-topk:], [cos_sims[k] for k in index[-topk:]]))

for i in index[-topk:]:    
    print("{}".format(mov_info[list(mov_feats.keys())[i]]))

###################### 检验效果

# 给定一个用户ID，找到评分最高的topk个电影
usr_a = 2022
topk = 194


# 获得ID为usr_a的用户评分过的电影及对应评分
rating_path = "./ratings.dat"
# 打开文件，ratings_data
with open(rating_path, 'r') as f:
    ratings_data = f.readlines()
    
usr_rating_info = {}
for item in ratings_data:
    item = item.strip().split("::")
    # 处理每行数据，分别得到用户ID，电影ID，和评分
    usr_id,movie_id,score = item[0],item[1],item[2]
    if usr_id == str(usr_a):
        usr_rating_info[movie_id] = float(score)

# 获得评分过的电影ID
movie_ids = list(usr_rating_info.keys())
print("ID为 {} 的用户，评分过的电影数量是: ".format(usr_a), len(movie_ids))


# 选出ID为usr_a评分最高的前topk个电影
ratings_topk = sorted(usr_rating_info.items(), key=lambda item:item[1])[-topk:]

movie_info_path = "./movies.dat"
# 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
with open(movie_info_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    
movie_info = {}
for item in data:
    item = item.strip().split("::")
    # 获得电影的ID信息
    v_id = item[0]
    movie_info[v_id] = item

for k, score in ratings_topk:
    print("{},{}".format(score, movie_info[k]))
