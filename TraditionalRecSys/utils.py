import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_data(file_dir):
    user_ids_dict, rated_item_ids_dict = {}, {}
    N, M, u_idx, i_idx = 0, 0, 0, 0
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()

        if int(u) not in user_ids_dict:
            user_ids_dict[int(u)] = u_idx
            u_idx += 1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)] = i_idx
            i_idx += 1
        data.append([user_ids_dict[int(u)], rated_item_ids_dict[int(i)], float(r)])

    f.close()
    N = u_idx
    M = i_idx

    return N, M, data, rated_item_ids_dict

#转换为用户评分矩阵
def sequence2mat(sequence, N, M):

    records_array = np.array(sequence)
    mat = np.zeros([N, M])
    row = records_array[:, 0].astype(int)
    col = records_array[:, 1].astype(int)
    values = records_array[:, 2].astype(np.float32)
    mat[row, col] = values

    return mat

def mae_rmse(r_pred, test_mat):
    y_pred = r_pred[test_mat > 0]
    y_true = test_mat[test_mat > 0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def evaluation(pred_mat, test_mat):
    mae, rmse = mae_rmse(pred_mat, test_mat)
    return mae, rmse

def figure(values_list, name='',color='g'):
    fig = plt.figure(name)
    x = range(len(values_list))
    plt.plot(x, values_list, color=color)
    plt.title(name + ' curve')
    plt.xlabel('Iterations')
    plt.ylabel(name)
    plt.show()