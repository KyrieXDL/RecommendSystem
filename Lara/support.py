import numpy as np
import pandas as pd
import evall

user_emb_matrix = np.array(pd.read_csv(r'./data/user_emb.csv', header=None))
user_attribute_matrix = np.array(pd.read_csv(r'./data/user_attribute.csv', header=None))
ui_matrix = np.array(pd.read_csv(r'./data/ui_matrix.csv', header=None))
user_emb_matrixT = np.transpose(user_attribute_matrix)

print(user_emb_matrixT.shape)
print(ui_matrix.shape)
def get_testdata():
    #测试集商品id
    test_item = np.array(pd.read_csv('./data/test_item.csv', header=None).astype(np.int32))
    #商品属性
    test_attribute = np.array(pd.read_csv('./data/test_attribute.csv', header=None).astype(np.int32))
    return test_item,test_attribute

#获取交互相似的k个用户
def get_intersection_similar_user(G_user, k):
    user_emb_matrixT = np.transpose(user_attribute_matrix)    
    A = np.matmul(G_user, user_emb_matrixT)   
    intersection_rank_matrix = np.argsort(-A)

    return intersection_rank_matrix[:, 0:k]

def test(test_item_batch, test_G_user):
    k_value = 20
    test_BATCH_SIZE = np.size(test_item_batch)
    #获取与生成器生成的虚拟用户相似的k个真实用户，作为商品的推荐用户
    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value)

    count = 0
    #根据user-item矩阵判断真实用户用户是否喜欢该商品
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist:
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    #k=20时的准确率
    p_at_20 = round(count/(test_BATCH_SIZE * k_value), 4)
           
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        RS.append(r)
    M_at_20 = evall.mean_average_precision(RS)
  
    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
    G_at_20 = ans/test_BATCH_SIZE

    '''k=10'''
    k_value = 10
    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist[:k_value]:
            
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    p_at_10 = round(count/(test_BATCH_SIZE * k_value), 4)
         
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        RS.append( r)
    M_at_10 = evall.mean_average_precision(RS)

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
    G_at_10 = ans/test_BATCH_SIZE

    return p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20


train = np.array(pd.read_csv('./data/train_data.csv', header =None))
def shuffle():
    np.random.shuffle(train)
    
def get_traindata(start_index, end_index):

    batch_data = train[start_index: end_index]

    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    real_user_emb_batch = user_emb_matrix[user_batch]
    
    return user_batch,item_batch,attr_batch,real_user_emb_batch


neg = np.array(pd.read_csv('./data/neg_data.csv', header =None))
def shuffle2():
    np.random.shuffle(neg)
    
def get_negdata(start_index, end_index):
    
    batch_data = neg[start_index: end_index]
    
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    real_user_emb_batch = user_emb_matrix[user_batch]

    return user_batch,item_batch,attr_batch,real_user_emb_batch




