import scipy.sparse as sp
import numpy as np
import random 
import sys
import json
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import repeat
from torch_geometric.data import NeighborSampler
from utils.hetero_data_loader import hetero_dl

def to_homogeneous(adj_list):
    """
    convert a list of adjacency matrices to a single homogeneous adjacency matrix
    """
    n = adj_list[0].shape[0]
    adj = sp.lil_matrix((n, n))
    for a in adj_list:
        adj += a
    adj[adj >= 1] = 1
    np.fill_diagonal(adj,1)
    return adj

def load_acm_raw(dataset, val_size=0.1,test_size=0.8):
    dataset_str = 'data/'+ dataset +'/'
    data_path = dataset_str + 'ACM.mat'

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    feat = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=int)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    # train-val-test: 0.1-0.1-0.8
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))

    val_idx = np.where(float_mask <= val_size)[0]
    train_idx = np.where((float_mask > val_size) & (float_mask <= 1-test_size))[0]
    test_idx = np.where(float_mask > 1-test_size)[0]

    adj_full = to_homogeneous([p_vs_a,p_vs_l])
    
    return adj_full, feat, labels, train_idx, val_idx, test_idx

def load_hetero_data(dataset, data_dir):
    dl = hetero_dl(data_dir + '/' +dataset)
    features = []
    max_dim = 0
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]).toarray())
            max_dim = dl.nodes['count'][i] if dl.nodes['count'][i] > max_dim else max_dim
        else:
            features.append(th)
            max_dim = th.shape[1] if th.shape[1] > max_dim else max_dim
    # pad zeros
    for i, feat in enumerate(features):
        if feat.shape[1] < max_dim:
            features[i] = np.hstack((feat,np.zeros((feat.shape[0], max_dim - feat.shape[1]))))
    features = np.concatenate(features,axis=0)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    adjM = adjM + adjM.T
    adjM[adjM > 1] = 1
    if dataset != 'IMDB':
        labels = labels.argmax(axis=1)
    labels = np.array(labels)
    return adjM, features, labels, train_idx, val_idx, test_idx


class DataGraph:
    '''
    Datasets used in GraphSAINT paper
    Heterogenous Datasets
    Synthetic Datasets
    '''

    def __init__(self, dataset, **kwargs):
        data_dir = 'data' if not 'data_dir' in kwargs else kwargs['data_dir'] 
        if dataset in ['ACM', 'DBLP', 'Freebase', 'IMDB']:
            # adj_full, feat, labels, idx_train, idx_val, idx_test = load_acm_raw(dataset)
            adj_full, feat, labels, idx_train, idx_val, idx_test = load_hetero_data(data_dir=data_dir, dataset=dataset)
            self.nnodes = adj_full.shape[0]
            self.nclass = len(np.unique(labels))
        else:
            dataset_str = data_dir + '/' + dataset +'/'
            adj_full = sp.load_npz(dataset_str+'adj_full.npz')
            self.nnodes = adj_full.shape[0]
            if dataset == 'ogbn-arxiv':
                adj_full = adj_full + adj_full.T
                adj_full[adj_full > 1] = 1

            role = json.load(open(dataset_str+'role.json', 'r'))
            idx_train = role['tr']
            idx_test = role['te']
            idx_val = role['va']

            if 'label_rate' in kwargs:
                label_rate = kwargs['label_rate']
                if label_rate < 1:
                    idx_train = idx_train[:int(label_rate*len(idx_train))]

            feat = np.load(dataset_str+'feats.npy')
            class_map = json.load(open(dataset_str + 'class_map.json', 'r'))
            labels = self.process_labels(class_map)

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        # ---- normalize feat ----
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        self.feat_train = feat[idx_train]
        self.feat_val = feat[idx_val]
        self.feat_test = feat[idx_test]

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.data_full = GraphData(
            adj_full, feat, labels, idx_train, idx_val, idx_test)
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_full
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        '''
        if using_half:
            vecs = vecs.half()
        '''
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)
        vecs = vecs.half()
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            class_ = list(budget.keys())[i]
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            #dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist, dim=-1) # include distance to all the other classes
            n_selected = (dist_<d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget[class_]] if len(rank) > budget[class_] else random.choices(rank, k=budget[class_])
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def get_cluster_centers(node_indexes, features, cluster_num):
        node_features = features[node_indexes]
        kmeans = KMeans(n_clusters=cluster_num,
                        random_state=0).fit(node_features)
        centers = kmeans.cluster_centers_
        center_ind = []
        for center in centers:
            dis = np.linalg.norm(node_features - center, axis=1)
            nearest_ind = np.argmin(dis)
            center_ind.append(node_indexes[nearest_ind])
        return center_ind

    def retrieve_class_sampler(self, c, adj, transductive, num=256, features=None, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[
                        self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        if features is None:
            batch = np.random.permutation(self.class_dict2[c])[:num]
        else:
            batch = self.get_cluster_centers(self.class_dict2[c], features, num)
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_sampler_val(self,transductive, num_per_class=64):
        #num = num_per_class*self.nclass
        if self.class_dict2 is None:
            self.class_dict2 = {}
            node_idx = []
            for i in range(self.nclass):
                if transductive:
                    idx_val = np.array(self.idx_val)
                    idx = idx_val[self.labels_val == i]
                else:
                    idx = np.arange(len(self.labels_val))[self.labels_val==i]
                self.class_dict2[i] = idx
                #node_idx.append(np.random.permutation(self.class_dict2[i])[:num_per_class])
                node_idx += np.random.permutation(self.class_dict2[i])[:num_per_class].tolist()
            self.class_dict2 = None
            return np.array(node_idx).reshape(-1)

    def retrieve_class_sampler_train(self,transductive, num_per_class=64):
        #num = num_per_class*self.nclass
        if self.class_dict2 is None:
            self.class_dict2 = {}
            node_idx = []
            for i in range(self.nclass):
                if transductive:
                    idx_val = np.array(self.idx_val)
                    idx = idx_val[self.labels_val == i]
                else:
                    idx = np.arange(len(self.labels_val))[self.labels_val==i]
                self.class_dict2[i] = idx
                #node_idx.append(np.random.permutation(self.class_dict2[i])[:num_per_class])
                node_idx += np.random.permutation(self.class_dict2[i])[:num_per_class].tolist()
            self.class_dict2 = None
            return np.array(node_idx).reshape(-1)

class GraphData:

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


class Data2Pyg:

    def __init__(self, data, device='cuda', transform=None, **kwargs):
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[
            0].to(device)
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[
            0].to(device)
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[
            0].to(device)
        self.nclass = data.nclass
        self.nfeat = data.nfeat
        self.class_dict = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (
                    self.data_train.y == i).cpu().numpy()
        idx = np.arange(len(self.data_train.y))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/'  # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                        slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass
