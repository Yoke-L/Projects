
import tensorflow as tf
import numpy as np
from model import model
# from model import model_q
import proto as pt


def train(n_way, n_shot, is_5shot):
    im_height, im_width, channels = 84, 84, 3
    dim = 64
    n_way = n_way
    n_shot = n_shot
    n_query = 15
    alpha = 10
    r = 0.01
    k = 0.1
    m = 5.

    lr = tf.placeholder(tf.float32)

    # support
    s = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    # query
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])

    s_shape = tf.shape(s)
    q_shape = tf.shape(q)
    num_classes, num_support = s_shape[0], s_shape[1]
    num_classes_q, num_queries = q_shape[0], q_shape[1]
    # labels
    y = tf.placeholder(tf.int64, [None, None])
    y_one_hot = tf.one_hot(y, depth=num_classes)
    emb_s = model(tf.reshape(s, [num_classes * num_support, im_height, im_width, channels]), dim)
    emb_q = model(tf.reshape(q, [num_classes_q * num_queries, im_height, im_width, channels]), dim, reuse=True)
    d=emb_q[3]

    def embs(x):
        a=(x[0]+x[4])/2
        b=(x[1]+x[5])/2
        c = (x[2] + x[6]) / 2
        d = (x[3] + x[7]) / 2
        return a,b,c,d

    emb_s = embs(emb_s)

    # def center(x):
    #     mean1 = tf.reduce_mean(x[0],axis=0)
    #     mean2 = tf.reduce_mean(x[1], axis=0)
    #     # mean3 = tf.reduce_mean(x[2], axis=0)
    #     # mean4 = tf.reduce_mean(x[3], axis=0)
    #     a = x[0] - mean1
    #     b = x[1] - mean2
    #     # c = x[2] - mean3
    #     # d = x[3] - mean4
    #     return a,b #,c,d
    #
    # emb_s = center(emb_s)

    # 1shot
    def centers(x):
        mean1 = tf.reduce_mean(x[0],axis=0)
        mean2 = tf.reduce_mean(x[1], axis=0)
        mean3 = tf.reduce_mean(x[2], axis=0)
        mean4 = tf.reduce_mean(x[3], axis=0)
        a = x[0] - mean1
        b = x[1] - mean2
        c = x[2] - mean3
        d = x[3] - mean4
        return a,b,c,d

    #emb_s = centers(emb_s)

    # def embs(x):
    #     a=(x[0]+x[2])/2
    #     b=(x[1]+x[3])/2
    #     return a,b
    #
    # emb_s = embs(emb_s)

    # def center(x):
    #     mean1 = tf.reduce_mean(x[0],axis=0)
    #     mean2 = tf.reduce_mean(x[1], axis=0)
    #     mean3 = tf.reduce_mean(x[2], axis=0)
    #     mean4 = tf.reduce_mean(x[3], axis=0)
    #     # mean5 = tf.reduce_mean(x[4], axis=0)
    #     # mean6 = tf.reduce_mean(x[5], axis=0)
    #     # mean7 = tf.reduce_mean(x[6], axis=0)
    #     # mean8 = tf.reduce_mean(x[7], axis=0)
    #     a = x[0] - mean1
    #     b = x[1] - mean2
    #     c = x[2] - mean3
    #     d = x[3] - mean4
    #     # a2 = y[4] - mean5
    #     # b2 = y[5] - mean6
    #     # c2 = y[6] - mean7
    #     # d2 = y[7] - mean8
    #     return a,b,c,d #,a2,b2,c2,d2
    #
    # emb_q = center(emb_q)


    # 1-shot
    def center(x):
        mean1 = tf.reduce_mean(x[0],axis=0)
        mean2 = tf.reduce_mean(x[1], axis=0)
        mean3 = tf.reduce_mean(x[2], axis=0)
        mean4 = tf.reduce_mean(x[3], axis=0)
        mean5 = tf.reduce_mean(x[4], axis=0)
        mean6 = tf.reduce_mean(x[5], axis=0)
        mean7 = tf.reduce_mean(x[6], axis=0)
        mean8 = tf.reduce_mean(x[7], axis=0)
        a = x[0] - mean1
        b = x[1] - mean2
        c = x[2] - mean3
        d = x[3] - mean4
        a2 = x[4] - mean5
        b2 = x[5] - mean6
        c2 = x[6] - mean7
        d2 = x[7] - mean8
        return a,b,c,d,a2,b2,c2,d2

    #emb_q = center(emb_q)

    def L2N(x):
        x_square = tf.square(x)
        x_sum = tf.reduce_sum(x_square, axis=1)
        x_norm = tf.sqrt(x_sum)
        x_norm = tf.reshape(x_norm, [-1,1])
        x = x/x_norm
        return x

    # def l2embs(x):
    #     a = L2N(x[0])
    #     b = L2N(x[1])
    #     return a,b
    #
    # def l2emb(x):
    #     a = L2N(x[0])
    #     b = L2N(x[1])
    #     c = L2N(x[2])
    #     d = L2N(x[3])
    #     return a,b,c,d
    #
    # emb_s = l2embs(emb_s)
    # emb_q = l2emb(emb_q)

    #1shot
    def l2embs(x):
        a = L2N(x[0])
        b = L2N(x[1])
        c = L2N(x[2])
        d = L2N(x[3])
        return a,b,c,d

    def l2emb(x):
        a = L2N(x[0])
        b = L2N(x[1])
        c = L2N(x[2])
        d = L2N(x[3])
        a2 = L2N(x[4])
        b2 = L2N(x[5])
        c2 = L2N(x[6])
        d2 = L2N(x[7])
        return a,b,c,d,a2,b2,c2,d2

    emb_s = l2embs(emb_s)
    emb_q = l2emb(emb_q)

    # if is_5shot:
    #     emb_s = pt.newprot(emb_s, n_shot, n_way)

    # if is_5shot:
    #     emb_s = pt.newprot(emb_s, n_shot, n_way)

    def emb_5(x, y,i,j):
        e_s = x[i]
        emb_dim = tf.shape(e_s)[-1]
        e_s = tf.reduce_mean(tf.reshape(e_s, [num_classes, num_support, emb_dim]), axis=1)
        e_q = y[j]
        d = pt.euclidean_distance(e_q, e_s)
        #d = pt.cos(e_q, e_s)
        #d=tf.sqrt(d)
        return d

    def emb_1(x, y, i, j):
        e_s = x[i]
        emb_dim = tf.shape(e_s)[-1]
        e_s = tf.reduce_mean(tf.reshape(e_s, [num_classes, num_support, emb_dim]), axis=1)
        e_q = y[j]
        d = pt.euclidean_distance(e_q, e_s)
        #d = pt.cos(e_q, e_s)
        d = tf.sqrt(d)
        return d

    if is_5shot:
        d1 = emb_5(emb_s, emb_q, 0, 0)
        d2 = emb_5(emb_s, emb_q, 1, 1)
        d3 = emb_5(emb_s, emb_q, 2, 2)
        d4 = emb_5(emb_s, emb_q, 3, 3)
        d5 = emb_5(emb_s, emb_q, 0, 4)
        d6 = emb_5(emb_s, emb_q, 1, 5)
        d7 = emb_5(emb_s, emb_q, 2, 6)
        d8 = emb_5(emb_s, emb_q, 3, 7)


    else:
        d1 = emb_1(emb_s, emb_q, 0, 0)
        d2 = emb_1(emb_s, emb_q, 1, 1)
        d3 = emb_1(emb_s, emb_q, 2, 2)
        d4 = emb_1(emb_s, emb_q, 3, 3)
        d5 = emb_1(emb_s, emb_q, 0, 4)
        d6 = emb_1(emb_s, emb_q, 1, 5)
        d7 = emb_1(emb_s, emb_q, 2, 6)
        d8 = emb_1(emb_s, emb_q, 3, 7)





    def sharpen(x):
        x = tf.pow(x, 2)
        x_sum = tf.reduce_sum(x,axis=1)
        x_sum = tf.reshape(x_sum,[-1,1])
        x = x / x_sum
        return x

    dists1 = d3+d4
    dists2 = d7+d8

    pro1 = tf.nn.softmax(-dists1)
    pro2 = tf.nn.softmax(-dists2)

    pro_ = (pro1+pro2)/2

    pro_ = tf.pow(pro_, 2)
    pro_sum = tf.reduce_sum(pro_, axis=1)
    pro_sum = tf.reshape(pro_sum, [-1, 1])
    pro_ = pro_/pro_sum

    y_one_hot_ = tf.add(tf.multiply(y_one_hot, (1 - r)), r / n_way)

    # 定义损失1
    pro_2 = tf.log(pro_)
    log_p_y = tf.reshape(pro_2, [num_classes_q, num_queries, -1])
    loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot_, log_p_y), axis=-1), [-1]))
    y_ = tf.reshape(y_one_hot, [-1, n_way])
    # regularization_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # # # 添加正则项
    # loss2 = regularization_losses + loss2

    # def triloss(x, i, m):
    #     d = x
    #     # y_ = tf.reshape(y_one_hot, [-1,n_way])
    #     alph_s = emb_s[i]
    #     emb_dim = tf.shape(alph_s)[-1]
    #     alph_s = tf.reduce_sum(tf.reshape(alph_s, [num_classes, num_support, emb_dim]), axis=1)
    #     d_alph = pt.euclidean_distance(alph_s, alph_s)
    #     #d_alph = pt.cos(alph_s, alph_s)
    #     #d_alph = tf.sqrt(d_alph)
    #     y_1000 = y_ * -2
    #     #y_1000 = y_ * 1000
    #     d_alph = tf.tile(d_alph, [1, n_query])
    #     d_alph = tf.reshape(d_alph, [n_way * n_query, n_way])
    #     d_alph = d_alph + y_1000
    #     #d_alph_min = tf.reduce_min(d_alph, axis=1)
    #     d_alph_min = tf.reduce_max(d_alph, axis=1)
    #     d_alph_min = tf.reshape(d_alph_min, [-1, 1])
    #     # d_alph = d_alph_m - y_1000
    #     #rate_s = d_alph - d_alph_min
    #     rate_s = d_alph_min-d_alph
    #     # rate_s = tf.tile(rate_s, [1, n_query])
    #     # rate_s = tf.reshape(rate_s, [n_way * n_query, n_way])
    #     rate_s = rate_s + 1
    #     rate_s = tf.log(rate_s)
    #     rate_s = rate_s + 1
    #     rate_s = rate_s - rate_s * y_
    #     m = rate_s * m
    #     # y_ = tf.reshape(y_one_hot, [-1, n_way])
    #     d_k = d * y_
    #     d_k = tf.reshape(tf.reduce_sum(d_k, axis=1), [-1, 1])
    #     #d_ik = -(d - d_k)
    #     d_ik = d - d_k
    #     d_ik = d_ik + m
    #     d_m = y_ * m
    #     d_ik = d_ik - d_m
    #     d_ik = tf.maximum(0., d_ik)
    #     #         d_k_ = d * y_
    #     #         d_k_ = tf.reduce_mean(tf.reduce_sum(d_k_, axis=1))
    #     loss = tf.reduce_mean(tf.reduce_sum(d_ik, axis=1))
    #     # loss = loss + d_k_
    #     return loss

    def triloss(x, i, m):
        d = x
        # y_ = tf.reshape(y_one_hot, [-1,n_way])
        alph_s = emb_s[i]

        emb_dim = tf.shape(alph_s)[-1]
        alph_s = tf.reduce_sum(tf.reshape(alph_s, [num_classes, num_support, emb_dim]), axis=1)
        d_alph = pt.euclidean_distance(alph_s, alph_s)
        #dd=d_alph
        y_1000 = y_ * 1000
        d_alph = tf.tile(d_alph, [1, n_query])
        d_alph = tf.reshape(d_alph, [n_way * n_query, n_way])
        d_alph = d_alph + y_1000

        d_alph_min = tf.reduce_min(d_alph, axis=1)
        d_alph_min = tf.reshape(d_alph_min, [-1, 1])
        # d_alph = d_alph_m - y_1000
        rate_s = d_alph/d_alph_min

        # rate_s = tf.tile(rate_s, [1, n_query])
        # rate_s = tf.reshape(rate_s, [n_way * n_query, n_way])
        # rate_s = rate_s + 1
        rate_s = tf.log(rate_s)
        rate_s = rate_s + 1
        rate_s = rate_s - rate_s * y_

        m = rate_s * m
        # y_ = tf.reshape(y_one_hot, [-1, n_way])
        d_k = d * y_
        d_k = tf.reshape(tf.reduce_sum(d_k, axis=1), [-1, 1])
        d_ik = -(d - d_k)
        d_ik = d_ik + m
        d_m = y_ * m
        d_ik = d_ik - d_m
        d_ik = tf.maximum(0., d_ik)
        #         d_k_ = d * y_
        #         d_k_ = tf.reduce_mean(tf.reduce_sum(d_k_, axis=1))
        loss = tf.reduce_mean(tf.reduce_sum(d_ik, axis=1))
        # loss = loss + d_k_
        return loss

    loss_t1 = triloss(d1, 0, 0.5)
    loss_t2 = triloss(d2, 1, 0.5)
    loss_t3 = triloss(d3, 2, 0.5)
    loss_t4 = triloss(d4, 3, 0.5)
    loss_t5 = triloss(d5, 0, 0.5)
    loss_t6 = triloss(d6, 1, 0.5)
    loss_t7 = triloss(d7, 2, 0.5)
    loss_t8 = triloss(d8, 3, 0.5)

    #loss_t = loss_t1+loss_t2+loss_t3 + loss_t4+loss_t5+loss_t6+loss_t7 + loss_t8
    loss_t = loss_t3+loss_t4+loss_t7+loss_t8
    loss = loss2 + loss_t*0.2
    #     y_ = tf.reshape(y_one_hot, [-1,n_way])
    #     dists_k = tf.multiply(y_, dists)
    #     dists_i = -(dists-m)
    #     dists_km = tf.multiply(y_,m)-dists_k
    #     dists_i = dists_i-dists_km
    #     dists_i = tf.maximum(0.,dists_i)
    #     loss_k = tf.reduce_mean(tf.reduce_sum(dists_k,axis=1))
    #     loss_i = tf.reduce_mean(tf.reduce_sum(dists_i,axis=1))
    #     loss = loss_k+loss_i

    # train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)

    # 计算准确率1
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # 参数初始化
    init = tf.global_variables_initializer()
    return init, train_op, acc, loss, s, q, y,lr, d



