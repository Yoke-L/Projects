import tensorflow as tf
import numpy as np


def conv(inputs, out_channels, name='down',trainable=True):
    with tf.variable_scope(name,regularizer=tf.contrib.layers.l2_regularizer(5e-4)):
        out = tf.layers.conv2d(inputs, out_channels, kernel_size=1, strides=1, trainable=trainable)
        out = tf.contrib.layers.batch_norm(out, updates_collections=None, decay=0.99, scale=True, center=True)
        out = tf.nn.relu(out)
        #out = tf.contrib.layers.max_pool2d(out, 2)
        return out


def conv_block(inputs, out_channels, name='block', trainable=True):
    with tf.variable_scope(name):
        shortcut = tf.layers.conv2d(inputs, out_channels, kernel_size=1, strides=1,trainable=trainable)
        shortcut = tf.contrib.layers.batch_norm(shortcut, updates_collections=None, decay=0.99, scale=True, center=True)

        conv1 = tf.layers.conv2d(inputs, out_channels, kernel_size=3, strides=1, padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1, updates_collections=None, decay=0.99, scale=True, center=True)
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, out_channels, kernel_size=3, strides=1, padding='SAME', trainable=trainable)
        conv2 = tf.contrib.layers.batch_norm(conv2, updates_collections=None, decay=0.99, scale=True, center=True)
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, out_channels, kernel_size=3, strides=1, padding='SAME', trainable=trainable)
        conv3 = tf.contrib.layers.batch_norm(conv3, updates_collections=None, decay=0.99, scale=True, center=True)
        conv3 = tf.nn.relu(conv3)

        conv3 = conv3 + shortcut
        conv3 = tf.nn.relu(conv3)

        out = tf.contrib.layers.max_pool2d(conv3, 2)

        return out


def res_net(x, h_dim, reuse=False):
    # 特征提取网络，在训练集上进行 64 way 预训练
    with tf.variable_scope('res', reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(5e-4)):
        net1 = conv_block(x, h_dim, name='block1', trainable=False)
        net2 = conv_block(net1, h_dim * 2, name='block2', trainable=False)
        net3 = conv_block(net2, h_dim * 4, name='block3', trainable=True)
        net4 = conv_block(net3, h_dim * 8, name='block4', trainable=True)
        out1 = net1
        out2 = net2
        out3 = net3
        out4 = net4
        out5 = net4
        net4 = tf.nn.avg_pool(net4, [1,5,5,1], [1,5,5,1], 'VALID')
        net4 = tf.reshape(net4, [-1, np.prod([int(dim) for dim in net4.get_shape()[1:]])])
        # out1 = tf.reduce_mean(net1, [1, 2])
        # out2 = tf.reduce_mean(net2, [1, 2])
        # 对浅层输出特征图进行下采样，与深层输出特征尺寸相同
        with tf.variable_scope('res5',reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(5e-4)):
            # out1 = conv(out1, h_dim*8, name='down1')
            # out2 = conv(out2, h_dim*8, name='down2')
            out3 = conv(out3, h_dim * 8, name='down3')
            #out4 = conv(out4, h_dim * 8, name='down4')

            # out1 = tf.contrib.layers.max_pool2d(out1, 2)
            # out2 = out2 + out1
            # out2 = tf.contrib.layers.max_pool2d(out2, 2)
            # out2 = tf.nn.relu(out2)
            # out3 = out3 + out2
            out3 = tf.contrib.layers.max_pool2d(out3, 2)
            out3 = tf.nn.relu(out3)
            out4 = out4 + out3
            out4 = tf.nn.relu(out4)
        return out1, out2, out3, out4


def pre_net(x):
    # 64 way 预测
    with tf.variable_scope('dense'):
        net = tf.layers.dense(inputs=x, units=64, activation=tf.nn.softmax, name='dense2', trainable=True)
        return net


def encoder(x,out_channels,reuse=False):
    # 嵌入模块，输入浅层特征与深层特征，权值共享，输出维度相同，便于不同嵌入空间之间的比较
    with tf.variable_scope('enc', reuse=reuse):
#         out = tf.layers.conv2d(x, out_channels*8, kernel_size=3,padding='SAME',strides=1)
#         out = tf.contrib.layers.batch_norm(out, updates_collections=None, decay=0.99, scale=True, center=True)
#         out = tf.nn.relu(out)
#         out = tf.layers.conv2d(x, out_channels*8, kernel_size=3, strides=1)
#         out = tf.contrib.layers.batch_norm(out, updates_collections=None, decay=0.99, scale=True, center=True)
#         out = tf.nn.relu(out)
#         out = tf.layers.conv2d(out, out_channels*4, kernel_size=1,padding='SAME', strides=1)
#         out = tf.contrib.layers.batch_norm(out, updates_collections=None, decay=0.99, scale=True, center=True)
#         out = tf.nn.relu(out)
#         out = tf.layers.conv2d(out, out_channels*8, kernel_size=1, strides=1)
#         out = tf.contrib.layers.batch_norm(out, updates_collections=None, decay=0.99, scale=True, center=True)
#         out = tf.nn.relu(out)
        #out = tf.contrib.layers.flatten(x)
        # 全局平均池化后，即为特征向量
        #out = tf.nn.avg_pool(x, [1,5,5,1], [1,5,5,1], 'VALID')
        out = tf.reduce_mean(x, [1, 2])
        #out = tf.reshape(out, [-1, np.prod([int(dim) for dim in out.get_shape()[1:]])])
        return out



