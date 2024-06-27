import torch
import numpy as np


class Base:

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.ipc = ipc
        np.random.seed(args.seed)
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

    def generate_labels_syn(self, data):
        from collections import Counter

        counter = Counter(data.labels_train)
        num_class_dict = {}
        # n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [
                len(labels_syn),
                len(labels_syn) + num_class_dict[c],
            ]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn


    def select(self):
        return


class KCenter(Base):

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        super(KCenter, self).__init__(data, args, ipc=ipc, device="cuda", **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            # dis = distance(feature, mean)[:,0]
            dis = torch.cdist(feature, mean)[:, 0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt - 1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)

            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)


class KMeans(Base):

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        super(KMeans, self).__init__(data, args, ipc=ipc, device="cuda", **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # k-means # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            feature = embeds[idx]

            # Randomly select initial centroids
            random_indices = np.random.choice(len(feature), size=cnt, replace=False)
            centroids = feature[random_indices]

            for _ in range(300):  # Maximum iterations for convergence
                # Compute distances to centroids
                distances = torch.cdist(feature, centroids)
                cluster_assignments = torch.argmin(distances, dim=1)

                # Update centroids
                for i in range(cnt):
                    assigned_points = feature[cluster_assignments == i]
                    if len(assigned_points) > 0:
                        centroids[i] = torch.mean(assigned_points, dim=0)

            # Randomly select nodes from each cluster
            for i in range(cnt):
                cluster_nodes = idx[(cluster_assignments == i).cpu()]
                if len(cluster_nodes) > 0:
                    selected_node = np.random.choice(
                        cluster_nodes, size=1, replace=False
                    )
                    idx_selected.append(selected_node)

        return np.hstack(idx_selected)


class Herding(Base):

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        super(Herding, self).__init__(data, args, ipc=ipc, device="cuda", **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        # herding # class by class
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean * (i + 1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)


class Center(Base):

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        super(Center, self).__init__(data, args, ipc=ipc, device="cuda", **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # center # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            feature = embeds[idx]
            centroid = torch.mean(feature, dim=0, keepdim=True)
            distances = torch.cdist(feature, centroid)[:, 0]
            centroid_index = torch.argmin(distances).item()
            idx_selected.extend([idx[centroid_index]] * cnt)

        return np.array(idx_selected)


class Random(Base):

    def __init__(self, data, args, ipc=False, device="cuda", **kwargs):
        super(Random, self).__init__(data, args, ipc=ipc, device="cuda", **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train

        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)
