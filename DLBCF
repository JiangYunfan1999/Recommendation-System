# 定义数据读取与处理类
class MovieLens(object):
    """数据读取与处理类
    user_info: 用户信息字典（user_id，age，job）
    movie_info: 电影信息字典（movie_id，title，category，years）
    rating_info: 评分字典（user_id:{movie_id: score}）
    dataset: 数据字典（user_info，movie_info，score）
    train_dataset: 80%训练集
    valid_dataset: 20%验证集
    """
    def __init__(self):
        # 声明每个数据文件的路径
        user_info_path = "./users.dat"
        rating_path = "./ratings.dat"
        movie_info_path = "./movies.dat"
        
        # 得到电影数据
        self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
        # 记录电影的最大ID
        self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat])
        self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title])
        self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
        
        # 记录用户数据的最大ID
        self.max_usr_id = 0
        self.max_usr_age = 0
        self.max_usr_job = 0
        # 得到用户数据
        self.user_info = self.get_user_info(user_info_path)
        
        # 得到评分数据
        self.rating_info = self.get_rating_info(rating_path)
        
        # 构建数据集 
        self.dataset = self.get_dataset(user_info=self.user_info,
                                        rating_info=self.rating_info,
                                        movie_info=self.movie_info)
        ######################################划分数据集，获得数据加载器#####################################
        self.train_dataset = self.dataset[:int(len(self.dataset)*0.8)]
        self.valid_dataset = self.dataset[int(len(self.dataset)*0.8):]
        print("Total Dataset Size: ", len(self.dataset))
        print("MovieLens dataset information: \n  User Number: {}\n"
              "  Movie Number: {}".format(len(self.user_info),len(self.movie_info)))
    
    # 得到电影数据
    def get_movie_info(self, path):
        # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中 
        with open(path, 'r', encoding="ISO-8859-1") as f:
            data = f.readlines()
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        movie_info, movie_titles, movie_cat = {}, {}, {}
        # 对电影名字、类别中不同的单词计数
        t_count, c_count = 1, 1

        count_tit = {}
        # 按行读取数据并处理
        for item in data:
            item = item.strip().split("::")
            v_id = item[0]
            v_title = item[1][:-7]
            cats = item[2].split('|')
            v_year = item[1][-5:-1]

            titles = v_title.split()
            # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
            for t in titles:
                if t not in movie_titles:
                    movie_titles[t] = t_count
                    t_count += 1
            # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
            for cat in cats:
                if cat not in movie_cat:
                    movie_cat[cat] = c_count
                    c_count += 1
            # 补0使电影名称对应的列表长度为15
            v_tit = [movie_titles[k] for k in titles]
            while len(v_tit)<15:
                v_tit.append(0)
            # 补0使电影种类对应的列表长度为6
            v_cat = [movie_cat[k] for k in cats]
            while len(v_cat)<6:
                v_cat.append(0)
            # 保存电影数据到movie_info中
            movie_info[v_id] = {'movie_id': int(v_id),
                                'title': v_tit,
                                'category': v_cat,
                                'years': int(v_year)}
        return movie_info, movie_cat, movie_titles

    # 得到用户数据
    def get_user_info(self, path):
        # 性别转换函数，M-0， F-1
        def gender2num(gender):
            return 1 if gender == 'F' else 0

        # 打开文件，读取所有行到data中
        with open(path, 'r') as f:
            data = f.readlines()
        # 建立用户信息的字典
        user_info = {}

        max_usr_id = 0
        #按行索引数据
        for item in data:
            # 去除每一行中和数据无关的部分
            item = item.strip().split("::")
            usr_id = item[0]
            # 将字符数据转成数字并保存在字典中
            user_info[usr_id] = {'user_id': int(usr_id),
                                'gender': gender2num(item[1]),
                                'age': int(item[2]),
                                'job': int(item[3])}
            self.max_usr_id = max(self.max_usr_id, int(usr_id))
            self.max_usr_age = max(self.max_usr_age, int(item[2]))
            self.max_usr_job = max(self.max_usr_job, int(item[3]))
        return user_info
    
    # 得到评分数据
    def get_rating_info(self, path):
        # 读取文件里的数据
        with open(path, 'r') as f:
            data = f.readlines()
        # 将数据保存在字典中并返回
        rating_info = {}
        for item in data:
            item = item.strip().split("::")
            usr_id,movie_id,score = item[0],item[1],item[2]
            if usr_id not in rating_info.keys():
                rating_info[usr_id] = {movie_id:float(score)}
            else:
                rating_info[usr_id][movie_id] = float(score)
        return rating_info
    
    # 构建数据集
    def get_dataset(self, user_info, rating_info, movie_info):
        trainset = []
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for movie_id in usr_ratings:
                trainset.append({'user_info': user_info[usr_id],
                                 'movie_info': movie_info[movie_id],
                                 'scores': usr_ratings[movie_id]})
        return trainset
    
    def load_data(self, dataset=None, mode='train'):
        # 定义数据迭代Batch大小
        BATCHSIZE = 256
        data_length = len(dataset)
        index_list = list(range(data_length))
        
        # 定义数据迭代加载器
        def data_generator():
            # 训练模式下，打乱训练数据
            if mode == 'train':
                random.shuffle(index_list)
            # 声明每个特征的列表
            usr_id_list,usr_gender_list,usr_age_list,usr_job_list = [], [], [], []
            mov_id_list,mov_tit_list,mov_cat_list = [], [], []
            score_list = []
            # 索引遍历输入数据集
            for idx, i in enumerate(index_list):
                # 获得特征数据保存到对应特征列表中
                usr_id_list.append(dataset[i]['user_info']['user_id'])
                usr_gender_list.append(dataset[i]['user_info']['gender'])
                usr_age_list.append(dataset[i]['user_info']['age'])
                usr_job_list.append(dataset[i]['user_info']['job'])

                mov_id_list.append(dataset[i]['movie_info']['movie_id'])
                mov_tit_list.append(dataset[i]['movie_info']['title'])
                mov_cat_list.append(dataset[i]['movie_info']['category'])
                mov_id = dataset[i]['movie_info']['movie_id']

                score_list.append(int(dataset[i]['scores']))
                
                # 如果读取的数据量达到当前的batch大小，就返回当前批次
                if len(usr_id_list)==BATCHSIZE:
                    # 转换列表数据为数组形式，reshape到固定形状
                    usr_id_arr = np.array(usr_id_list)
                    usr_gender_arr = np.array(usr_gender_list)
                    usr_age_arr = np.array(usr_age_list)
                    usr_job_arr = np.array(usr_job_list)

                    mov_id_arr = np.array(mov_id_list)
                    mov_cat_arr = np.reshape(np.array(mov_cat_list), [BATCHSIZE, 6]).astype(np.int64)
                    mov_tit_arr = np.reshape(np.array(mov_tit_list), [BATCHSIZE, 1, 15]).astype(np.int64)

                    scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)

                    # 返回当前批次数据
                    yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], \
                           [mov_id_arr, mov_cat_arr, mov_tit_arr], scores_arr

                    # 清空数据
                    usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
                    mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []
                    
        return data_generator
     
     
# 数据读取与处理
dataset = MovieLens()

# 重新划分数据集
dataset.train_dataset = [dataset.dataset[i] for i in train_idx]
dataset.valid_dataset = [dataset.dataset[i] for i in test_idx]

# 定义数据读取器
train_loader = dataset.load_data(dataset=dataset.train_dataset, mode='train')

# 迭代读取数据， Batchsize = 256
for idx, data in enumerate(train_loader()):
    usr, mov, score = data
    print("用户属性的数据维度：\nuser_id，gender，age，job")
    for v in usr:
        print(v.shape, end='   ')
    print("\n电影属性的数据维度：\nmovie_id，title，category")
    for v in mov:
        print(v.shape, end='   ')
    break

# 数据集划分结果
print("Training set: ",len(dataset.train_dataset))
print("Validation set: ",len(dataset.valid_dataset))

# 使用余弦相似度
def similarity(feature_1, feature_2):
    res = F.common.cosine_similarity(feature_1, feature_2)
    return res
    
# 定义模型设计类
class Model(paddle.nn.Layer):
    def __init__(self, use_mov_title, use_mov_cat, use_age_job, fc_sizes):
        super(Model, self).__init__()
        
        # 将传入的name信息和bool型参数添加到模型类中
        self.use_mov_title = use_mov_title
        self.use_mov_cat = use_mov_cat
        self.use_usr_age_job = use_age_job
        self.fc_sizes=fc_sizes
        
        # 获取数据集的信息，并构建训练和验证集的数据迭代器
        Dataset = MovieLens()
        self.Dataset = Dataset
        self.trainset = self.Dataset.train_dataset
        self.valset = self.Dataset.valid_dataset
        self.train_loader = self.Dataset.load_data(dataset=self.trainset, mode='train')
        self.valid_loader = self.Dataset.load_data(dataset=self.valset, mode='valid')
        
        usr_embedding_dim=32
        gender_embeding_dim=16
        age_embedding_dim=16
        
        job_embedding_dim=16
        mov_embedding_dim=16
        category_embedding_dim=16
        title_embedding_dim=32
        
        """ 用户特征提取的神经网络 """
        USR_ID_NUM = Dataset.max_usr_id + 1
        
        # 对用户ID做映射，并紧接着一个Linear层
        self.usr_emb = Embedding(num_embeddings=USR_ID_NUM, embedding_dim=usr_embedding_dim, sparse=False)
        self.usr_fc = Linear(in_features=usr_embedding_dim, out_features=32)
        
        # 对用户性别信息做映射，并紧接着一个Linear层
        USR_GENDER_DICT_SIZE = 2
        self.usr_gender_emb = Embedding(num_embeddings=USR_GENDER_DICT_SIZE, embedding_dim=gender_embeding_dim)
        self.usr_gender_fc = Linear(in_features=gender_embeding_dim, out_features=16)
        
        # 对用户年龄信息做映射，并紧接着一个Linear层
        USR_AGE_DICT_SIZE = Dataset.max_usr_age + 1
        self.usr_age_emb = Embedding(num_embeddings=USR_AGE_DICT_SIZE, embedding_dim=age_embedding_dim)
        self.usr_age_fc = Linear(in_features=age_embedding_dim, out_features=16)
        
        # 对用户职业信息做映射，并紧接着一个Linear层
        USR_JOB_DICT_SIZE = Dataset.max_usr_job + 1
        self.usr_job_emb = Embedding(num_embeddings=USR_JOB_DICT_SIZE, embedding_dim=job_embedding_dim)
        self.usr_job_fc = Linear(in_features=job_embedding_dim, out_features=16)
        
        # 新建一个Linear层，用于整合用户数据信息
        self.usr_combined = Linear(in_features=80, out_features=200)
        
        """ 电影特征提取的神经网络 """
        # 对电影ID信息做映射，并紧接着一个Linear层
        MOV_DICT_SIZE = Dataset.max_mov_id + 1
        self.mov_emb = Embedding(num_embeddings=MOV_DICT_SIZE, embedding_dim=mov_embedding_dim)
        self.mov_fc = Linear(in_features=mov_embedding_dim, out_features=32)
       
        # 对电影类别做映射
        CATEGORY_DICT_SIZE = len(Dataset.movie_cat) + 1
        self.mov_cat_emb = Embedding(num_embeddings=CATEGORY_DICT_SIZE, embedding_dim=category_embedding_dim, sparse=False)
        self.mov_cat_fc = Linear(in_features=category_embedding_dim, out_features=32)
        
        # 对电影名称做映射
        MOV_TITLE_DICT_SIZE = len(Dataset.movie_title) + 1
        self.mov_title_emb = Embedding(num_embeddings=MOV_TITLE_DICT_SIZE, embedding_dim=title_embedding_dim, sparse=False)
        self.mov_title_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2,1), padding=0)
        self.mov_title_conv2 = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding=0)
        
        # 新建一个FC层，用于整合电影特征
        self.mov_concat_embed = Linear(in_features=96, out_features=200)
        
        """ 捕获合并后向量的深层语义信息 """
        # 用户特征的后续全连接层
        user_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._user_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=user_sizes[i],
                out_features=user_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(user_sizes[i]))))
            self.add_sublayer('linear_user_%d' % i, linear)
            self._user_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('user_act_%d' % i, act)
                self._user_layers.append(act)
        
        # 电影特征的后续全连接层，不共享参数
        movie_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._movie_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=movie_sizes[i],
                out_features=movie_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(movie_sizes[i]))))
            self.add_sublayer('linear_movie_%d' % i, linear)
            self._movie_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('movie_act_%d' % i, act)
                self._movie_layers.append(act)
        
    # 定义计算用户特征的前向运算过程
    def get_usr_feat(self, usr_var):
        """ 提取用户特征 """
        # 获取到用户数据
        usr_id, usr_gender, usr_age, usr_job = usr_var
        # 将用户的ID数据经过embedding和Linear计算，得到的特征保存在feats_collect中
        feats_collect = []
        usr_id = self.usr_emb(usr_id)
        usr_id = self.usr_fc(usr_id)
        usr_id = F.relu(usr_id)
        feats_collect.append(usr_id)
        
        # 计算用户的性别特征，并保存在feats_collect中
        usr_gender = self.usr_gender_emb(usr_gender)
        usr_gender = self.usr_gender_fc(usr_gender)
        usr_gender = F.relu(usr_gender)
        feats_collect.append(usr_gender)
        # 计算用户的年龄-职业特征，并保存在feats_collect中
        if self.use_usr_age_job:
            # 计算用户的年龄特征，并保存在feats_collect中
            usr_age = self.usr_age_emb(usr_age)
            usr_age = self.usr_age_fc(usr_age)
            usr_age = F.relu(usr_age)
            feats_collect.append(usr_age)
            # 计算用户的职业特征，并保存在feats_collect中
            usr_job = self.usr_job_emb(usr_job)
            usr_job = self.usr_job_fc(usr_job)
            usr_job = F.relu(usr_job)
            feats_collect.append(usr_job)
        
        # 将用户的特征级联，并通过Linear层得到最终的用户特征
        usr_feat = paddle.concat(feats_collect, axis=1)
        user_features = F.tanh(self.usr_combined(usr_feat))
        
        #通过n层全链接层，获得用于计算相似度的用户特征和电影特征
        for n_layer in self._user_layers:
            user_features = n_layer(user_features)

        return user_features

    # 定义电影特征的前向计算过程
    def get_mov_feat(self, mov_var):
        """ 提取电影特征 """
        # 获得电影数据
        mov_id, mov_cat, mov_title = mov_var
        feats_collect = []
        # 获得batchsize的大小
        batch_size = mov_id.shape[0]
        # 计算电影ID的特征，并存在feats_collect中
        mov_id = self.mov_emb(mov_id)
        mov_id = self.mov_fc(mov_id)
        mov_id = F.relu(mov_id)
        feats_collect.append(mov_id)
        
        # 计算电影种类的映射
        if self.use_mov_cat:
            # 计算电影种类的特征映射，对多个种类的特征求和得到最终特征
            mov_cat = self.mov_cat_emb(mov_cat)
            mov_cat = paddle.sum(mov_cat, axis=1, keepdim=False)

            mov_cat = self.mov_cat_fc(mov_cat)
            feats_collect.append(mov_cat)
            
        # 计算电影名字的映射
        if self.use_mov_title:
            # 计算电影名字的特征映射，对特征映射使用卷积计算最终的特征
            mov_title = self.mov_title_emb(mov_title)
            mov_title = F.relu(self.mov_title_conv2(F.relu(self.mov_title_conv(mov_title))))
            mov_title = paddle.sum(mov_title, axis=2, keepdim=False)
            mov_title = F.relu(mov_title)
            mov_title = paddle.reshape(mov_title, [batch_size, -1])
            feats_collect.append(mov_title)
            
        # 使用一个全连接层，整合所有电影特征，映射为一个200维的特征向量
        mov_feat = paddle.concat(feats_collect, axis=1)
        mov_features = F.tanh(self.mov_concat_embed(mov_feat))

        for n_layer in self._movie_layers:
            mov_features = n_layer(mov_features)

        return mov_features
    
    # 定义推荐算法的前向计算
    def forward(self, usr_var, mov_var):
        # 计算用户特征和电影特征
        usr_feat = self.get_usr_feat(usr_var)
        mov_feat = self.get_mov_feat(mov_var)
        ######################### 计算特征的相似度 ################################
        sim = similarity(usr_feat, mov_feat).reshape([-1, 1])
        # 将相似度扩大范围到和电影评分相同数据范围
        res = paddle.scale(sim, scale=5)
        
        return usr_feat, mov_feat, res

# 定义模型训练函数
def train(model, lr=0.001, epoch=10):
    # 配置训练参数
    lr = lr
    Epoches = epoch
    paddle.set_device('cpu') 

    # 启动训练
    model.train()
    # 获得数据读取器
    data_loader = model.train_loader
    # 使用adam优化器，学习率使用0.01
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    
    records_list = []
    for epoch in range(0, Epoches):
        for idx, data in enumerate(data_loader()):
            # 获得数据，并转为tensor格式
            usr, mov, score = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]
            scores_label = paddle.to_tensor(score)
            # 计算出算法的前向计算结果
            _, _, scores_predict = model(usr_v, mov_v)
            # 计算loss
            loss = F.square_error_cost(scores_predict, scores_label)
            avg_loss = paddle.mean(loss)
            
            records_list.append(avg_loss)
            if idx % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))
                
            # 损失函数下降，并清除梯度
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        # 每个epoch 保存一次模型
        paddle.save(model.state_dict(), './checkpoint/epoch'+str(epoch)+'.pdparams')
    
    return np.array(records_list)

# 设定参数
fc_sizes=[128, 64, 32]
use_mov_title, use_mov_cat, use_age_job = True, True, True
# 启动训练
model = Model(use_mov_title, use_mov_cat, use_age_job, fc_sizes)
records_dl = train(model, lr=0.001, epoch=5)

# 绘制Loss
df = pd.DataFrame(records_dl[:10000],columns=['Loss'])
df['iteration'] = np.arange(0,10000)
sns.lineplot(data=df,x='iteration',y='Loss')
    
plt.title('Loss curve of DLBCF')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# 定义模型评估函数
def evaluation(model, params_file_path):
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    mae_set = []
    rmse_set = []
    acc_set = []
    # 遍历每条评分记录
    for idx, data in enumerate(model.valid_loader()):
        usr, mov, score_label = data
        usr_v = [paddle.to_tensor(var) for var in usr]
        mov_v = [paddle.to_tensor(var) for var in mov]
        # 返回预测评分
        _, _, scores_predict = model(usr_v, mov_v)
        pred_scores = scores_predict.numpy()
        # 单个样本Absolute Error
        mae_set.append(np.abs(pred_scores - score_label))
        # 单个样本Squared Error
        rmse_set.append(np.abs(pred_scores - score_label)**2)
       # 单个样本是否正确
        acc_set.append(np.abs(pred_scores - score_label) < 0.5)
    
    mae = np.mean(mae_set)
    rmse = np.sqrt(np.mean(rmse_set))
    acc = np.mean(acc_set)
    return mae, rmse, acc

param_path = "./checkpoint/epoch"

epoch = 5

for i in range(epoch):
    MAE, RMSE, ACC = evaluation(model, param_path+str(i)+'.pdparams')
    print("Epoch "+str(i))
    print("  MAE: %.4f, RMSE: %.4f, ACC: %.4f"
          %(MAE, RMSE, ACC))
          
          
# 定义特征保存函数
def get_usr_mov_features(model, params_file_path):
    paddle.set_device('cpu') 
    usr_feat_dic = {}
    mov_feat_dic = {}
    
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
        usr_info, mov_info, score = dataset[i]['user_info'], dataset[i]['movie_info'],dataset[i]['scores']
        usrid = str(usr_info['user_id'])
        movid = str(mov_info['movie_id'])

        # 获得用户数据，计算得到用户特征，保存在字典中
        if usrid not in usr_feat_dic.keys():
            usr_id_v = list2tensor(usr_info['user_id'], [1])
            usr_age_v = list2tensor(usr_info['age'], [1])
            usr_gender_v = list2tensor(usr_info['gender'], [1])
            usr_job_v = list2tensor(usr_info['job'], [1])

            usr_in = [usr_id_v, usr_gender_v, usr_age_v, usr_job_v]
            usr_feat = model.get_usr_feat(usr_in)

            usr_feat_dic[usrid] = usr_feat.numpy()
        
        # 获得电影数据，计算得到电影特征，保存在字典中
        if movid not in mov_feat_dic.keys():
            mov_id_v = list2tensor(mov_info['movie_id'], [1])
            mov_tit_v = list2tensor(mov_info['title'], [1, 1, 15])
            mov_cat_v = list2tensor(mov_info['category'], [1, 6])

            mov_in = [mov_id_v, mov_cat_v, mov_tit_v]
            mov_feat = model.get_mov_feat(mov_in)

            mov_feat_dic[movid] = mov_feat.numpy()
    
    usr_feat_list = []
    mov_feat_list = []
    for i in range(len(usr_feat_dic)):
        feat = list(usr_feat_dic.values())[i].reshape(-1)
        usr_feat_list.append(feat)
    for i in range(len(mov_feat_dic)):
        feat = list(mov_feat_dic.values())[i].reshape(-1)
        mov_feat_list.append(feat)
    
    return usr_feat_list, mov_feat_list


param_path = "./checkpoint/epoch1.pdparams"
user_features, movie_features = get_usr_mov_features(model, param_path)  

pd.DataFrame(user_features).to_csv("user_features_32.csv", index=0)
pd.DataFrame(movie_features).to_csv("movie_features_32.csv", index=0)
print("User/Movie Features Saved!")
