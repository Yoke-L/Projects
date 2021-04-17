import tensorflow as tf
import numpy as np
from run import run


# Implement details
# query: 15
# train: 20-way 5-shot  val/test: 5-way 5-shot
# train: 5-way 5-shot   val/test: 5-way 5-shot
# train: 20-way 1-shot  val/test: 5-way 1-shot
# train: 5-way  1-shot  val/test: 5-way 1-shot
# train/val/test: 20-way 5-shot
# train/val/test: 20-way 1-shot

train_data_all = np.load('few-shot-train.npz')
val_data_all = np.load('few-shot-val.npz')
test_data_all = np.load('few-shot-test.npz')

train_data = train_data_all['features']
val_data = val_data_all['features']
test_data = test_data_all['features']
print(train_data.shape)
train_y=train_data_all['targets']

def aug(x):
    a=[]
    for i in range(x.shape[0]):
        a.append(x[i,::-1,:,:])
    a=np.array(a).reshape(-1,84*84*3)
    return a

aug_x = aug(train_data)
t1=train_data.reshape(-1,84*84*3)
t2=np.concatenate((t1,aug_x),axis=1)
t2=t2.reshape(-1,84*84*3)
t2=t2.reshape(64,-1,84,84,3)
print(t2.shape)

train_data = train_data.reshape(64,600,84,84,3)
val_data = val_data.reshape(16,600,84,84,3)
test_data = test_data.reshape(20,600,84,84,3)




def train_5shot(n_way,n_shot,is_5shot,n_query,t_n_way, t_n_shot, t_n_query, train_data=train_data, val_data=val_data, test_data=test_data):
    run(n_way,n_shot,is_5shot,n_query,t_n_way, t_n_shot, t_n_query, train_data, val_data, test_data)


def train_1shot(n_way,n_shot,is_5shot,n_query,t_n_way, t_n_shot, t_n_query, train_data=t2, val_data=val_data, test_data=test_data):
    run(n_way,n_shot,is_5shot,n_query,t_n_way, t_n_shot, t_n_query, train_data, val_data, test_data)


if __name__ == '__main__':
    train_5shot(n_way=5,
               n_shot=5,
               is_5shot=True,
               n_query=15,
               t_n_way=5,
               t_n_shot=5,
               t_n_query=15)

    # train_1shot(n_way=5,
    #             n_shot=1,
    #             is_5shot=False,
    #             n_query=15,
    #             t_n_way=5,
    #             t_n_shot=1,
    #             t_n_query=15)