def mae_rmse_acc(r_pred,test_mat):
    """定义评价指标计算函数：
    1. 输入预测评分矩阵和真实评分矩阵
    2. 输出评分预测的误差
    """
    # 对于存在用户评分
    y_pred = r_pred[test_mat>0]
    y_true = test_mat[test_mat>0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    acc = np.mean(np.abs(r_pred-test_mat)<0.5)
    return mae, rmse, acc

# 矩阵分解的模型设计
class MF():
    def __init__(self, train_list, test_list, N, M, 
                 D=10, learning_rate=0.001,lambda_regularizer=0.1, max_iteration=50):
        self.train_list = train_list
        self.test_list = test_list
        self.M = M   # 电影个数
        self.N = N   # 用户个数
        self.D = D   # 隐变量个数
        self.learning_rate = learning_rate   # 梯度下降的学习率
        self.lambda_regularizer = lambda_regularizer    # 正则项参数
        self.max_iteration = max_iteration    # 最大迭代次数
    
    # 训练模型，得到用户特征矩阵P、电影特征矩阵Q、每次更新的模型评估指标
    def train(self):
        P = np.random.normal(0, 0.1, (self.N, self.D))
        Q = np.random.normal(0, 0.1, (self.M, self.D))
        
        train_mat = list2matrix(self.train_list, N=self.N, M=self.M)
        test_mat = list2matrix(self.test_list, N=self.N, M=self.M)
        
        # 记录每次迭代训练的模型评估指标
        records_list = []
        for step in range(self.max_iteration):
            los = 0.0
            for data in self.train_list:
                u,i,r = data
                # 对每个评分，梯度下降更新参数
                P[u],Q[i],ls = self.update(P[u], Q[i], r=r,
                                           learning_rate=self.learning_rate,
                                           lambda_regularizer = self.lambda_regularizer)
                # 加总训练集上的Loss
                los += ls
            # 当前参数计算预测评分
            pred_mat = self.prediction(P, Q)
            # 计算当前模型的评估指标
            mae, rmse,acc = mae_rmse_acc(pred_mat, test_mat)
            # 记录本次模型表现
            records_list.append(np.array([los,mae,rmse,acc]))
            
            if step % 10 == 0:
                print('Step:%d \n  Loss:%.4f, MAE:%.4f, RMSE:%.4f, ACC:%.4f'
                      %(step,los,mae,rmse,acc))
        print('End. \n Loss:%.4f, MAE:%.4f, RMSE:%.4f, ACC:%.4f'
              %(records_list[-1][0],records_list[-1][1],records_list[-1][2],records_list[-1][3]))
        
        return P, Q, np.array(records_list)
    
    # 梯度下降更新
    def update(self, p, q, r, learning_rate, lambda_regularizer):
        error = r - np.dot(p, q.T)
        p += learning_rate*(error*q - lambda_regularizer*p)
        q += learning_rate*(error*p - lambda_regularizer*q)
        # 优化目标（损失函数）
        loss = 0.5*(error**2 + lambda_regularizer*(np.sum(np.square(p)) + np.sum(np.square(q))))
        return p, q, loss
    
    # 计算预测评分
    def prediction(self, P, Q):
        N,D = P.shape
        M,D = Q.shape
        
        rating_list = []
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)  # 某一个用户对所有电影的预测评分
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
        
        
# 矩阵分解参数设定
D = 10
learning_rate = 0.005
lambda_regularizer = 0.1
max_iteration = 20

# 构建模型
mf_model = MF(train_list=train_ratings_list, test_list=test_ratings_list, N=N, M=M,
              D=D, learning_rate=learning_rate, lambda_regularizer=lambda_regularizer, max_iteration=max_iteration)

# 训练模型
P, Q, records_mf = mf_model.train()

# 绘制Loss
#fig = plt.figure(name)
#x = range(len(values))
#plt.plot(x, values, color='blue', linewidth=3)

df = pd.DataFrame(records_mf,columns=['Loss','MAE','RMSE','ACC'])
df['iteration'] = np.arange(0,len(records_mf))
sns.lineplot(data=df,x='iteration',y='Loss')
    
plt.title('Loss curve of MFBCF')
plt.xticks(np.arange(0,21,1))
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# 模型评估
last_iter = 5

for i in range(last_iter):
    i += 1
    print("Iteration "+str(i))
    print("  MAE: %.4f, RMSE: %.4f, ACC: %.4f"
          %(records_mf[-i,1],records_mf[-i,2],records_mf[-i,3]))
          

# 保存隐因子矩阵
pd.DataFrame(P).to_csv("user_latent_factors_10.csv", index=0)
pd.DataFrame(Q).to_csv("movie_latent_factors_10.csv", index=0)
print("User/Movie Latent Factors Saved!")
