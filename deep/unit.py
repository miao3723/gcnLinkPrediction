from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from deep.model import GCNModel
from deep.optimizer import Optimizer



# Set random seed
seed = 200
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()  # 转换存储格式，稀疏矩阵
    coords = np.vstack((sparse_mx.row,
                        sparse_mx.col)).transpose()  # https://blog.csdn.net/csdn15698845876/article/details/73380803 #位置
    values = sparse_mx.data  # 值
    shape = sparse_mx.shape  # 形状
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)  # 稀疏化存储
    adj_ = adj + sp.eye(adj.shape[0])  # 加上对角线上的连接
    rowsum = np.array(adj_.sum(1))  # 每行相加
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 对角线为度矩阵对角线开平方分之一
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})  # 空的单位矩阵的另外一种存储 feed_dict是给placeholder提供值
    feed_dict.update({placeholders['adj']: adj_normalized})  # 跟度矩阵处理后的值
    feed_dict.update({placeholders['adj_orig']: adj})  # 邻近矩阵
    return feed_dict


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


'''
Title：gcn
url：
     http://snap.stanford.edu/deepnetbio-ismb/
     http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html
Type:  Protein/drug/RNA/…. （研究具体内容）
Description: 模块主要是输入m*m的矩阵，通过GCN学习到m*n的拓扑特征，输出新的打分矩阵np.dot(emb, emb.T) 
Input Args: adj 邻接矩阵 m*m feature_mat特征矩阵，拼接好的m*k的特征矩阵，不输入则默认为one-hot编码
Output Shape: 新的打分矩阵 m*m
'''
def get_new_scoring_matrices(adj,feature_mat=None):
    adj=sp.csr_matrix(adj) #转换为稀疏矩阵
    num_nodes = adj.shape[0]  # 点的个数
    num_edges = adj.sum() #边的数量
    # Featureless
    if feature_mat==None:
        features = sparse_to_tuple(sp.identity(num_nodes))  #
    else:
        features=sparse_to_tuple(feature_mat)
    num_features = features[2][1]  # 列数
    features_nonzero = features[1].shape[0]  # data的形状，也还是节点数
    adj_orig = adj - sp.dia_matrix((adj.diagonal(), [0]), shape=adj.shape)  # 把临近矩阵的对角线1元素去掉
    adj_orig.eliminate_zeros()  # 不存储值为0的元素
    adj_norm = preprocess_graph(adj) #normlize D-1/2AD-/2

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    #GCN模型
    model = GCNModel(placeholders, num_features, features_nonzero, name='yeast_gcn')

    # Create optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            num_nodes=num_nodes,
            num_edges=num_edges,
        )


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    adj_label = adj+ sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

    print('Optimization Finished!')
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    adj_rec = sigmoid(np.dot(emb, emb.T))
    return adj_rec



