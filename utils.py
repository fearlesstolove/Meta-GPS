import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from dgl.data import CoraFullDataset
from sklearn import preprocessing
from torch_geometric.datasets import LastFMAsia, WikipediaNetwork, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.utils as tgu

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset_source):
    n1s = []
    n2s = []
    for line in open("few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s),max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                 shape=(num_nodes, num_nodes))

    data_train = sio.loadmat("few_shot_data/{}_train.mat".format(dataset_source))
    train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))

    data_test = sio.loadmat("few_shot_data/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))

    labels = np.zeros((num_nodes,1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    #degree = np.sum(adj, axis=1)
    #degree = torch.FloatTensor(degree)

    adj_noloop = normalize_adj(adj) # useless
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)

    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_noloop = sparse_mx_to_torch_sparse_tensor(adj_noloop).to(device) # useless
    
    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class, adj_noloop


def load_coradata():
    print("this is CoraFull")
    random.seed(12)
    data = CoraFullDataset()
    minus_node = [1, 4, 43, 68, 69]  # node number less than 70
    g = data[0]
    features = g.ndata['feat']
    label = g.ndata['label']
    np_label = label.numpy()

    #degree = g.in_degrees()
    #degree = torch.FloatTensor(degree.numpy())
    adj = g.adjacency_matrix(scipy_fmt='coo')
    adj_noloop = normalize_adj(adj)  # useless
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_noloop = sparse_mx_to_torch_sparse_tensor(adj_noloop)  # useless

    class_list = []
    for cla in np_label:
        if cla not in class_list:
            class_list.append(cla)

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(np_label):
        id_by_class[cla].append(id)

    class_train = random.sample(class_list, 55)
    class_test = list(set(class_list).difference(set(class_train)))
    class_valid = random.sample(class_train, 15)
    class_train = list(set(class_train).difference(set(class_valid)))

    # minus less number node
    class_train = list(set(class_train).difference(set(minus_node)))
    class_valid = list(set(class_valid).difference(set(minus_node)))
    class_test = list(set(class_test).difference(set(minus_node)))

    #class_train = [0, 2, 3, 9, 12, 17, 18, 19, 20, 23, 25, 28, 30, 31, 32, 34, 37, 38, 39, 42, 44, 45, 46, 47, 48, 51, 52, 53, 54, 57, 58, 59, 61, 62, 63, 64, 67]
    #class_valid = [65, 66, 6, 7, 8, 40, 11, 15, 16, 49, 22, 24, 27, 60]
    #class_test = [21, 56, 13, 35, 26, 41, 55, 10, 5, 36, 33, 50, 29, 14]

    return adj, features, label, class_train, class_valid, class_test, id_by_class, adj_noloop


def load_intrisitic_data(dataset, seed):
    print("dataset is {}".format(dataset))
    if dataset == 'CoraFull':
        return load_coradata()
    else:
        #random.seed(seed) change
        if dataset == 'lastfm':
            lastfm = LastFMAsia("few_shot_data/lastfm")
            lastfm_dataset = lastfm[0]
            label = lastfm_dataset.y
            features = lastfm_dataset.x
            edge = lastfm_dataset.edge_index
            train_num = 12
            valid_num = 6
            '''
            class_train = [0, 2, 5, 6, 8, 9]
            class_valid = [15, 3, 10, 11, 7, 4]
            class_test = [1, 12, 13, 14, 16, 17]
            '''
        elif dataset == 'reddit':
            reddit = Reddit("few_shot_data/reddit")
            #reddit = Reddit2("few_shot_data/reddit")
            reddit_dataset = reddit[0]
            label = reddit_dataset.y
            features = reddit_dataset.x
            edge = reddit_dataset.edge_index
            train_num = 31
            valid_num = 10
        elif dataset == 'ogb-arxiv':
            arxiv = PygNodePropPredDataset(name='ogbn-arxiv', root="few_shot_data")
            arxiv_dataset = arxiv[0]
            label = arxiv_dataset.y
            label = label.reshape(1, -1)[0]
            features = arxiv_dataset.x
            edge = arxiv_dataset.edge_index
            train_num = 28
            valid_num = 12
            '''
            class_train = [2, 35, 36, 5, 6, 37, 11, 17, 19, 21, 22, 25, 27, 28, 29, 30]
            class_valid = [34, 24, 20, 0, 4, 1, 23, 31, 14, 7, 15, 3]
            class_test = [32, 33, 38, 39, 8, 9, 10, 12, 13, 16, 18, 26]
            '''

        elif dataset == 'ogb-product':
            product = PygNodePropPredDataset('ogbn-products', root="few_shot_data")
            product_dataset = product[0]
            label = product_dataset.y
            label = label.reshape(1, -1)[0]
            features = product_dataset.x
            edge = product_dataset.edge_index
            train_num = 35
            valid_num = 12
            class_train = [5, 6, 8, 9, 13, 15, 18, 21, 22, 24, 25, 26, 28, 29, 30, 31, 36, 38, 41, 42, 43]
            class_valid = [20, 19, 37, 23, 12, 27, 0, 14, 3, 16]
            class_test = [32, 1, 2, 34, 4, 7, 10, 11, 44, 17]

        elif dataset == 'ccs':
            chameleon_dataset = WikipediaNetwork('few_shot_data/wiki', "chameleon")
            chameleon = chameleon_dataset[0]
            src, dst = chameleon.edge_index
            g1 = dgl.graph((src, dst))
            g1.ndata['fea'] = chameleon.x
            g1.ndata['label'] = chameleon.y

            crocodile_dataset = WikipediaNetwork('few_shot_data/wiki', 'crocodile')
            crocodile = crocodile_dataset[0]
            src, dst = crocodile.edge_index
            g2 = dgl.graph((src, dst))
            g2.ndata['fea'] = crocodile.x
            g2.ndata['label'] = crocodile.y + 5

            squirrel_dataset = WikipediaNetwork('few_shot_data/wiki', 'squirrel')
            squirrel = squirrel_dataset[0]
            src, dst = squirrel.edge_index
            g3 = dgl.graph((src, dst))
            g3.ndata['fea'] = squirrel.x
            g3.ndata['label'] = squirrel.y + 10

            g = dgl.batch([g1, g2, g3])
            features = g.ndata['fea']
            label = g.ndata['label']
            adj = g.adjacency_matrix(scipy_fmt='coo')
            train_num = 8
            valid_num = 0

        np_label = label.numpy().flatten().tolist()
        if dataset != 'ccs':
            adj = tgu.to_scipy_sparse_matrix(edge)
        adj_noloop = normalize_adj(adj)  # useless
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_noloop = sparse_mx_to_torch_sparse_tensor(adj_noloop)  # useless

        class_list = []
        for cla in np_label:
            if cla not in class_list:
                class_list.append(cla)
        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(np_label):
            id_by_class[cla].append(id)

        if dataset != 'ogb-product':
            class_train = random.sample(class_list, train_num)
            class_test = list(set(class_list).difference(set(class_train)))
            class_valid = random.sample(class_train, valid_num)
            class_train = list(set(class_train).difference(set(class_valid)))

        return adj, features, label, class_train, class_valid, class_test, id_by_class, adj_noloop


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def task_generator(id_by_class, class_list, n_way, k_shot, m_query, task_num):

    x_spt = {} # [0: array(1,2,5), 1: array(2,4,5)]
    x_qry = {}
    class_selected = {}  # {0: [1, 3, 5], 1: [2, 5, 6]}
    # sample class indices
    for i in range(task_num):
        class_selected[i] = random.sample(class_list, n_way)
        id_support = []
        id_query = []
        for cla in class_selected[i]:
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        x_spt[i] = np.array(id_support)
        x_qry[i] = np.array(id_query)

    return x_spt, x_qry, class_selected




