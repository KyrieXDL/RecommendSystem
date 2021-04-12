import numpy as np
import random
from TraditionalRecSys.utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class pmf():
    def __init__(self, train_data, test_data, N, M, K=10, learning_rate=0.001, lamda_regularizer=0.1, max_iteration=50):
        self.train_data = train_data
        self.test_data = test_data
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration

    def train(self):
        #初始化，P，Q矩阵，满足正态分布
        P = np.random.normal(0, 0.1, (self.N, self.K))
        Q = np.random.normal(0, 0.1, (self.M, self.K))

        train_mat = sequence2mat(sequence=self.train_data, N=self.N, M=self.M)
        test_mat = sequence2mat(sequence=self.test_data, N=self.N, M=self.M)

        records_list = []
        for epoch in range(self.max_iteration):
            loss = 0.0
            cnt = 0
            for data in self.train_data:
                u, i, r = data
                P[u], Q[i], ls = self.update(P[u], Q[i], r=r,
                                             learning_rate=self.learning_rate,
                                             lamda_regularizer=self.lamda_regularizer)
                loss += ls
                cnt += 1
            loss /= cnt
            pred_mat = self.prediction(P, Q)
            mae, rmse = evaluation(pred_mat, test_mat)
            records_list.append(np.array([loss, mae, rmse]))

            if epoch % 5 == 0:
                print('epoch %d:  loss:%.4f,mae:%.4f,rmse:%.4f'% (epoch,loss, mae, rmse))

        print(' train finished. \n loss:%.4f,mae:%.4f,rmse:%.4f'% (records_list[-1][0], records_list[-1][1], records_list[-1][2]))
        figure(np.array(records_list)[:,0],'loss','r')
        figure(np.array(records_list)[:,1],'mae','b')
        figure(np.array(records_list)[:,2],'rmse','g')
        return P, Q, np.array(records_list)

    def update(self, p, q, r, learning_rate=0.001, lamda_regularizer=0.1):
        error = r - np.dot(p, q.T)
        p = p + learning_rate * (error * q - lamda_regularizer * p)
        q = q + learning_rate * (error * p - lamda_regularizer * q)
        loss = 0.5 * (error ** 2 + lamda_regularizer * (np.square(p).sum() + np.square(q).sum()))
        return p, q, loss

    def prediction(self, P, Q):
        N, K = P.shape
        M, K = Q.shape

        rating_list = []
        for u in range(N):
            u_rating = np.sum(P[u, :] * Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred

if __name__ == '__main__':
    #u.data的数据格式： user_id, item_id, rating, timestamp
    N,M,data,_ = load_data('./data/ml-100k/u.data') #N: 用户数量， M: 电影数量, data：评分
    print(np.shape(data))

    model = pmf(data,data,N=N,M=M,K=10,max_iteration=50)
    model.train()