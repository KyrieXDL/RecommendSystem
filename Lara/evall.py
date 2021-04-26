import numpy as np
def RR(r, k) :
    for i in range(k):
        if r[i] == 1:
            return 1.0/(i + 1.0)
    return 0

def dcg_at_k(r, k, method=1):
    """
    dcg
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """
    ndcg
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max   
    
def precision_at_k(r, k):
    """
    r: 被推荐商品i的用户列表，1表示用户真实会买，0表示不会买即错误推荐
    返回前k的准确率
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    """
    返回前k准确率的均值
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])



    



   
    