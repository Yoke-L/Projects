import tensorflow as tf
import numpy as np
import train3 as t
from dataload import load_dataset
from dataload2 import load_dataset2
import os
import tensorflow.contrib.slim as slim


n_episodes = 100000

#l = 0.0001
im_height, im_width, channels = 84, 84, 3
train_classes = 64
val_classes = 16
test_classes = 20


def run(n_way, n_shot, is_5shot, n_query, t_n_way, t_n_shot, t_n_query, train_data, val_data, test_data):
    init, train_op, acc, loss, s, q, y, lr,d = t.train(n_way, n_shot, is_5shot)
    # 保存特征提取网络的预训练参数
    include = ['down1', 'down2', 'res5']
    variables = tf.contrib.framework.get_variables_to_restore()
    variables_to_resotre = [v for v in variables if
                            (v.name.split('/')[0] == 'res' and v.name.split('/')[1] not in include)]
    saver = tf.train.Saver(variables_to_resotre)
    saver2 = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as sess:
        sess.run(init)
        model_dir = './ckpt/'
        model_name = 'fsl_330_.ckpt'

        saver.restore(sess, os.path.join(model_dir, model_name))
    #saver2.restore(sess, './ckpt_1/fsl_6000_acc_0.72282.ckpt')
        l = 0.001
        for epi in range(n_episodes):

            if (epi + 1) % 1000 == 0:
                l *= 0.8
            # if l < 1e-6:
            #     l = 0.0005
            support, query, labels = load_dataset2(n_way, n_shot, n_query, train_classes, train_data)
            _, ls, ac = sess.run([train_op, loss, acc], feed_dict={s: support, q: query, y: labels, lr: l})
            # dd = sess.run(d, feed_dict={s: support, q: query, y: labels, lr: l})
            # print(dd)
            if (epi + 1) % 50 == 0:
                    print('[episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_episodes, ls, ac))
            #             if (epi+1)%1000==0:
            #                 vacc=0.
            #                 for val_epi in range(600):
            #                     v_support, v_query, v_labels = load_dataset(t_n_way, t_n_shot, t_n_query, val_classes, val_data)
            #                     ls, ac = sess.run([loss, acc], feed_dict={s: v_support, q: v_query, y: v_labels})
            #                     vacc+=ac
            #                 vacc/=600
            #                 print('val_acc = {:.5f}'.format(vacc))
            #                 if is_5shot:
            #                     saver.save(sess, save_path='ckpt_5shot/fsl_{}_acc_{:.5f}.ckpt'.format(epi + 1, vacc))
            #                 else:
            #                     saver.save(sess, save_path='ckpt_1shot/fsl_{}_acc_{:.5f}.ckpt'.format(epi + 1, vacc))
            if (epi + 1) % 500 == 0:
                    tacc = 0.
                    for test_epi in range(600):
                        t_support, t_query, t_labels = load_dataset2(t_n_way, t_n_shot, t_n_query, test_classes, test_data)
                        ls, ac = sess.run([loss, acc], feed_dict={s: t_support, q: t_query, y: t_labels})
                        tacc += ac
                    tacc /= 600
                    print('test_acc = {:.5f}'.format(tacc))
                    if is_5shot:
                        saver2.save(sess, save_path='ckpt_2/fsl_{}_acc_{:.5f}.ckpt'.format(epi + 1, tacc))
                    else:
                        saver2.save(sess, save_path='ckpt_5/fsl_{}_acc_{:.5f}.ckpt'.format(epi + 1, tacc))



