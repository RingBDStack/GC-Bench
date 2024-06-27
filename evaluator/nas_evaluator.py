from torch_geometric.data import InMemoryDataset, Data
import scipy.sparse as sp

class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process____(self):
        dpr_data = self.dpr_data
        try:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero().cpu()).cuda().T
        except:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero()).cuda()
        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def process(self):
        dpr_data = self.dpr_data
        if type(dpr_data.adj) == torch.Tensor:
            adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
            edge_index_selfloop = adj_selfloop.nonzero().T
            edge_index = edge_index_selfloop
            edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        else:
            adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
            edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
            edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()

        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels


        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        if self.len() == 1:
            return copy.copy(self._data)
        
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data
        

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

