import tensorflow as tf
import numpy as np
from encoder4 import encoder
from encoder4 import res_net

def model(x, dim, reuse=False):
    # 对支持集数据进行图像变换
    x2 = x[:, :, ::-1, :]
    out1_1, out1_2,out1_3, out1_4= res_net(x,  dim, reuse=reuse)
    out2_1, out2_2,out2_3, out2_4= res_net(x2,  dim, reuse=True)

    out1_1 = encoder(out1_1, dim, reuse=reuse)
    out1_2 = encoder(out1_2, dim, reuse=True)
    out1_3 = encoder(out1_3, dim, reuse=True)
    out1_4 = encoder(out1_4, dim, reuse=True)

    out2_1 = encoder(out2_1, dim, reuse=True)
    out2_2 = encoder(out2_2, dim, reuse=True)
    out2_3 = encoder(out2_3, dim, reuse=True)
    out2_4 = encoder(out2_4, dim, reuse=True)
    return out1_1, out1_2, out1_3, out1_4,out2_1, out2_2,out2_3, out2_4

# def model(x, dim, reuse=False):
#     # 对支持集数据进行图像变换
#     x2 = x[:, :, ::-1, :]
#     x3 = x[:, ::-1, :, :]
#     o1, out1_1, out1_2, out1_3, out1_4, outt1 = res_net(x,  dim, reuse=reuse)
#     o2, out2_1, out2_2, out2_3, out2_4, outt2 = res_net(x2,  dim, reuse=True)
#     o3, out3_1, out3_2, out3_3, out3_4, outt3 = res_net(x3, dim, reuse=True)
#
#     out1_1 = encoder(out1_1, dim, reuse=reuse)
#     out1_2 = encoder(out1_2, dim, reuse=True)
#     out1_3 = encoder(out1_3, dim, reuse=True)
#     out1_4 = encoder(out1_4, dim, reuse=True)
#     out2_1 = encoder(out2_1, dim, reuse=True)
#     out2_2 = encoder(out2_2, dim, reuse=True)
#     out2_3 = encoder(out2_3, dim, reuse=True)
#     out2_4 = encoder(out2_4, dim, reuse=True)
#     out3_1 = encoder(out3_1, dim, reuse=True)
#     out3_2 = encoder(out3_2, dim, reuse=True)
#     out3_3 = encoder(out3_3, dim, reuse=True)
#     out3_4 = encoder(out3_4, dim, reuse=True)
#
#     return out1_1, out1_2, out1_3, out1_4, out2_1, out2_2, out2_3, out2_4, out3_1, out3_2,out3_3,out3_4,outt1, outt2,outt3


# def model_q(x, dim, reuse=False):
#     # 查询集数据不进行变换
#     o1, out1_1, out1_2 = res_net(x, dim, reuse=reuse)

#     out1_1 = encoder(out1_1, dim, reuse=reuse)
#     out1_2 = encoder(out1_2, dim, reuse=True)

#     return out1_1, out1_2

# def model_2(x, reuse=False):
#     with tf.variable_scope('encoder', reuse=reuse):
#         x2 = x[:, :, ::-1, :]
#         x3 = x[:, ::-1, :, :]
#         out1_1, out1_2 = dense_encoder(x)
#         out2_1, out2_2 = dense_encoder(x2)
#         out3_1, out3_2 = dense_encoder(x3)
#         return out1_1, out1_2, out2_1, out2_2, out3_1, out3_2




