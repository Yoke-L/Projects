import numpy as np
# miniImageNet
def DataLoad(n_way,n_shot,n_query,classes,data):
    im_height, im_width, channels = 84, 84, 3
    data = data
    epi_classes = np.random.permutation(classes)[:n_way]
    support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)
    query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(600)[:n_shot + n_query]
        support[i] = data[epi_cls, selected[:n_shot]]
        query[i] = data[epi_cls, selected[n_shot:]]
    labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
    return support, query, labels


# 5-way 5-shot
def train_5_5(load_data):
    return DataLoad(n_way=5, n_shot=5, n_query=15, classes=64, data=load_data)
def val_5_5(load_data):
    return DataLoad(n_way=5, n_shot=5, n_query=15, classes=16, data=load_data)
def test_5_5(load_data):
    return DataLoad(n_way=5, n_shot=5, n_query=15, classes=20, data=load_data)

# 5-way 1-shot
def train_5_1(load_data):
    return DataLoad(n_way=5, n_shot=1,n_query=15, classes=64, data=load_data)
def val_5_1(load_data):
    return DataLoad(n_way=5, n_shot=1, n_query=15, classes=16, data=load_data)
def test_5_1(load_data):
    return DataLoad(n_way=5, n_shot=1, n_query=15, classes=20, data=load_data)

# 20-way 5-shot
def train_20_5(load_data):
    return DataLoad(n_way=20, n_shot=5, n_query=15, classes=64, data=load_data)
def val_20_5(load_data):
    return DataLoad(n_way=20, n_shot=5, n_query=15, classes=16, data=load_data)
def test_20_5(load_data):
    return DataLoad(n_way=20, n_shot=5, n_query=15, classes=20, data=load_data)

# 20-way 1-shot
def train_20_1(load_data):
    return DataLoad(n_way=20, n_shot=1, n_query=15, classes=64, data=load_data)
def val_20_1(load_data):
    return DataLoad(n_way=20, n_shot=1, n_query=15, classes=16, data=load_data)
def test_20_1(load_data):
    return DataLoad(n_way=20, n_shot=1, n_query=15, classes=20, data=load_data)

def load_dataset2(n_way,n_shot,n_query,classes,data):
    return DataLoad(n_way, n_shot, n_query, classes, data)

# if __name__ == '__main__':
#     a,b=train_5_5()
#     print(b.shape)