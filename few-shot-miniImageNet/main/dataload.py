import numpy as np

def DataLoad(n_way,n_shot,n_query,classes,data):
    im_height, im_width, channels = 84, 84, 3
    data = data
    epi_classes = np.random.permutation(classes)[:n_way]
    support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)
    query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(1200)[:n_shot + n_query]
        support[i] = data[epi_cls, selected[:n_shot]]
        query[i] = data[epi_cls, selected[n_shot:]]
    labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
    return support, query, labels


def load_dataset(n_way, n_shot, n_query, classes,data):
    return DataLoad(n_way, n_shot, n_query, classes, data)

