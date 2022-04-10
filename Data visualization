### 所有评分的分布情况：
# 已做userID和movieID标准化的评分数据
ratings = pd.DataFrame(ratings_list)
ratings.columns = ['idx','user','movie','score']
ratings = ratings.iloc[:,1:]

ratings['score'].mean()

sns.histplot(data=ratings, x='score', shrink=20)
plt.title('Distribution of scores')

### 用户观看过的电影数量：
temp_df = ratings.groupby('user').count()
# 描述统计
temp_df['movie'].describe()
sns.histplot(data=temp_df, x='movie')
plt.title("Number of movie watched")

### 影片被观看的用户人数：
temp_df = ratings.groupby('movie').count()
# 描述统计
temp_df['user'].describe()
sns.histplot(data=temp_df, x='user')
plt.title("Number of users watched")

### 用户平均打分：
temp_df = ratings.groupby('user').mean()
sns.histplot(data=temp_df, x='score')
plt.title("Average score of users")
plt.xlabel('average score')

### 用户的兴趣多样性：
# 选出某个用户评分=5的电影
def usr_liked_mov(usr_id, mov_id):
    mov_info = {}
    mov_info_path =  "movies.dat"
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item
    print("User:", usr_id)
    print("Movie List:")
    
    for id in mov_id:
        print("Movie:", id, mov_info[str(id)])

usr_id = 1956
usr_watched_list = ratings[ratings['user']==usr_id]

usr_liked_idx = usr_watched_list[usr_watched_list['score']==5]['movie']
usr_liked = [movie_id_dict[i] for i in usr_liked_idx]

usr_watched_idx = usr_watched_list['movie']
usr_watched = [movie_id_dict[i] for i in usr_watched_idx]

usr_liked_mov(usr_id, usr_watched)
usr_liked_mov(usr_id, usr_liked)
