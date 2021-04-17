import tensorflow as tf
import numpy as np

def euclidean_distance(a, b):
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_sum(tf.square(a - b), axis=2)


def cos(a, b):
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    sum = tf.reduce_sum(a*b, axis=-1)
    #sum=tf.reshape(sum,[N,M,1])
    a_sum = tf.reduce_sum(tf.square(a),axis=-1)
    a_sum = tf.sqrt(a_sum)
    #a_sum = tf.reshape(a_sum,[N,M,1])
    b_sum = tf.reduce_sum(tf.square(b), axis=-1)
    b_sum = tf.sqrt(b_sum)
    #b_sum = tf.reshape(b_sum, [N, M, 1])
    return sum/(a_sum*b_sum)

def normalization2(x):
    x_max = tf.reduce_max(x,axis=1)
    x_min = tf.reduce_min(x,axis=1)
    x_mm = x_max-x_min
    x_mm = tf.reshape(x_mm,[-1,1])
    x_min = tf.reshape(x_min,[-1,1])
    x=(x-x_min)/x_mm
    return x

def normalization(x):
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    x=(x-x_min)/(x_max-x_min)
    return x

# 加权原型-距离归一化
# def newprot(x,n_shot,n_way):
#     e=x
#     b=[]
#     n=n_way
#     count = 0
#     for j in range(2):
#         emb1 = e[j+count]
#         count+=1
#         emb2 = e[j+count]
#         for i in range(n):
#             # 同一层的特征图映射到同一嵌入空间，即原图与变换后的图在同一空间
#             a1 = emb1[i*n_shot:(i+1)*n_shot]
#             a2 = emb2[i*n_shot:(i+1)*n_shot]
#             a = tf.concat(a1, a2, axis=0)
#             # 计算同一空间的类内距离，邻接矩阵
#             D = euclidean_distance(a, a)
#             # 按行求和，即类内每一个数据点与其他数据点距离之和，返回一个列表
#             D1 = tf.reduce_sum(D,1)
#             # D_mean = tf.reduce_mean(D1)
#             # D1 = D1/D_mean
#             # 为了便于计算，将各距离和归一化
#             D1 = normalization(D1)
#             #D2=tf.reduce_sum(D,[0,1])/n
#             D2 = []
#             for i_s in range(n_shot):
#                 D2.append(tf.exp(-D1[i_s]))
#             D2 = tf.reduce_sum(D2)
#             for i_c in range(n_shot):
#                 # 前 n_shot 为原图权重，后 n_shot 为变换图权重
#                 b.append(emb1[i * n_shot + i_c] * (tf.exp(-D1[i_c])) / D2)
#                 b.append(emb2[i * n_shot + i_c] * (tf.exp(-D1[i_c + n_shot])) / D2)
#     # 前为浅层特征信息，后为深层特征信息
#     c = b[:n * n_shot*2]
#     d = b[n * n_shot*2:]
#     c = tf.stack(c)
#     d = tf.stack(d)
#     return c, d

def newprot(x,n_shot,n_way):
    e=x
    b=[]
    n=n_way
    for j in range(4):
        emb = e[j]
        for i in range(n):
            a=emb[i*n_shot:(i+1)*n_shot]
            D=euclidean_distance(a, a)
            D1=tf.reduce_sum(D,1)
            # D_mean = tf.reduce_mean(D1)
            # D1 = D1/D_mean
            #D1=normalization(D1)
            #D2=tf.reduce_sum(D,[0,1])/n
            D2 = []
            for i_s in range(n_shot):
                D2.append(tf.exp(-D1[i_s]))
            D2 = tf.reduce_sum(D2)
            for i_c in range(n_shot):
                b.append(emb[i*n_shot+i_c]*(tf.exp(-D1[i_c]))/D2)
    c = b[:n * n_shot]
    d = b[n * n_shot:2 * (n * n_shot)]
    c2 = b[2 * (n * n_shot):3 * (n * n_shot)]
    d2 = b[3 * (n * n_shot):]
    c = tf.stack(c)
    d = tf.stack(d)
    c2 = tf.stack(c2)
    d2 = tf.stack(d2)
    return c, d, c2, d2

# def newprot(x,n_shot,n_way):
#     e=x
#     b=[]
#     n=n_way
#     for j in range(8):
#         emb = e[j]
#         for i in range(n):
#             a=emb[i*n_shot:(i+1)*n_shot]
#             D=euclidean_distance(a, a)
#             D1=tf.reduce_sum(D,1)
#             # D_mean = tf.reduce_mean(D1)
#             # D1 = D1/D_mean
#             D1=normalization(D1)
#             #D2=tf.reduce_sum(D,[0,1])/n
#             D2 = []
#             for i_s in range(n_shot):
#                 D2.append(tf.exp(-D1[i_s]))
#             D2 = tf.reduce_sum(D2)
#             for i_c in range(n_shot):
#                 b.append(emb[i*n_shot+i_c]*(tf.exp(-D1[i_c]))/D2)
#     c = b[:n * n_shot]
#     d = b[n * n_shot:2 * (n * n_shot)]
#     c2 = b[2 * (n * n_shot):3 * (n * n_shot)]
#     d2 = b[3 * (n * n_shot):4 * (n * n_shot)]
#     c3 = b[4 * (n * n_shot):5 * (n * n_shot)]
#     d3 = b[5 * (n * n_shot):6 * (n * n_shot)]
#     c4 = b[6 * (n * n_shot):7 * (n * n_shot)]
#     d4 = b[7 * (n * n_shot):]
#     c = tf.stack(c)
#     d = tf.stack(d)
#     c2 = tf.stack(c2)
#     d2 = tf.stack(d2)
#     c3 = tf.stack(c3)
#     d3= tf.stack(d3)
#     c4 = tf.stack(c4)
#     d4 = tf.stack(d4)
#     return c, d, c2, d2, c3, d3, c4, d4

# 加权原型-距离不归一化处理
# def newprot(x,n_shot,n_way):
#     e=x
#     b=[]
#     n=n_way
#     for j in range(4):
#         emb = e[j]
#         for i in range(n):
#             a=emb[i*n_shot:(i+1)*n_shot]
#             D=euclidean_distance(a, a)
#             D1=tf.reduce_sum(D,1)
#             # D_mean = tf.reduce_mean(D1)
#             # D1 = D1/D_mean
#             #D1=normalization(D1)
#             #D2=tf.reduce_sum(D,[0,1])/n
#             #D2 = []
#             # for i_s in range(n_shot):
#             #     D2.append(tf.exp(-D1[i_s]))
#             D2 = tf.reduce_sum(D1)
#             for i_c in range(n_shot):
#                 b.append(emb[i*n_shot+i_c]*(D1[i_c])/D2)
#     c = b[:n * n_shot]
#     d = b[n * n_shot:2 * (n * n_shot)]
#     c2 = b[2 * (n * n_shot):3 * (n * n_shot)]
#     d2 = b[3 * (n * n_shot):]
#     c = tf.stack(c)
#     d = tf.stack(d)
#     c2 = tf.stack(c2)
#     d2 = tf.stack(d2)
#     return c, d, c2, d2