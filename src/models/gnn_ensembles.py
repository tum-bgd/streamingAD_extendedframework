import os
from typing import Callable
import numpy as np
import pandas as pd
from typing_extensions import override
from copy import copy, deepcopy
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from dataRepresentation import WindowStreamVectors
from modelWrapper import ModelWrapper
from nonconformity_scores.nonconformity_wrapper import calc_nonconformity_scores


class EnsembleGNNWrapper(ModelWrapper):
    def __init__(self, publisher: WindowStreamVectors, subscribers: list, model_id: str,
                 model_type: str, window_length: int, ensemble_length: int, model=None, save_paths=None, batch_size=32, debug=False) -> None:
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.dataset_train = None
        self.window_length = window_length
        self.nodes_between_length = 800
        self.ensemble_length = ensemble_length
        self.edge_indices_list = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.performance_counters = np.zeros((ensemble_length))
        self.save_paths = save_paths
        self.batch_size = batch_size
        self.debug = debug

        self._sample_edge_indices_by_routes()

    def _sample_edge_indices(self):
        lam = 3
        self.edge_indices_list = []
        for i in range(self.ensemble_length):
            edges = []
            for j in range(0, self.window_length):
                edges.append((j, j))
            for j in range(0, self.window_length-1):
                edges.append((j, j+1))
            for j in range(0, self.window_length):
                start, end = j+2, self.window_length-2
                # num_potential_edges = (1/2) * (self.window_length - 1) * (self.window_length - 2)
                num_potential_edges = end - start
                for k in range(start, end):
                    if np.random.random() < np.exp(-lam * (end - k) / num_potential_edges):
                        edges.append((j, k))
            # for j in range(0, self.window_length):
            #     for k in range(0, self.window_length)
            #         if np.random.random() < np.exp(-lam * (self.window_length - k)): # / num_potential_edges:
            #             edges.append((j, k))
            self.edge_indices_list.append(torch.tensor([
                [edges[j][0] for j in range(len(edges))],
                [edges[j][1] for j in range(len(edges))]
            ]))
            
    def _sample_edge_indices_by_routes(self):
        input_len = output_len = 100
        between_len= 100
        routes_per_component = 30
        component_target_len = 10
        max_route_len = 20
        start_nodes = np.arange(0, input_len)
        end_nodes = np.arange(input_len + between_len, input_len + between_len + output_len)
        self.edge_indices_list = []
        for comp_idx in range(self.ensemble_length):
            comp_start_idx = comp_idx * int(input_len / self.ensemble_length)
            comp_start_nodes = start_nodes[comp_start_idx : comp_start_idx + component_target_len]
            comp_end_nodes = end_nodes[comp_start_idx : comp_start_idx + component_target_len]
            edges = []
            for i in range(routes_per_component):
                nodes_idx = np.random.randint(0, len(comp_start_nodes))
                j = comp_start_nodes[nodes_idx]
                k = 0
                comp_offset = comp_idx * int(between_len / self.ensemble_length)
                edges.append((j, j + comp_offset))
                j += comp_offset
                k += 1
                while j < comp_end_nodes[nodes_idx]:
                    draw = np.random.random()
                    next_node_offset = int(1 + -np.ceil(np.log2(draw)))
                    next_node_offset *= 1
                    edges.append((j, min(j + next_node_offset, comp_end_nodes[nodes_idx])))
                    j += next_node_offset
                    k += 1
                    if k == max_route_len:
                        edges.append((j, comp_end_nodes[nodes_idx]))
                        break
            self.edge_indices_list.append(torch.tensor([
                [edges[j][0] for j in range(len(edges))],
                [edges[j][1] for j in range(len(edges))]
            ]))
    
    def _calc_number_of_routes(self, dist):
        return 1 + sum([2**(dist - 1 - j) for j in range(dist-1, 0, -1)])

    def _performance_weights(self):
        return F.softmax(torch.tensor(self.performance_counters), dim=0)

    @override
    def train(self, x: np.ndarray, epochs):
        if self.edge_indices_list is not None:
            data_list = []
            for i in range(len(x)):
                kwargs = {}
                for j in range(self.ensemble_length):
                    kwargs[f'inp_{j}'] = torch.from_numpy(
                        np.concatenate([x[i].astype(np.float32), np.zeros((self.nodes_between_length + self.window_length, x[i].shape[1]), dtype=np.float32)]))
                    kwargs[f'edge_index_{j}'] = self.edge_indices_list[j]
                data_list.append(EnsembleGNNData(
                    ensemble_length=self.ensemble_length, **kwargs))

            loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True, follow_batch=[
                                f'inp_{i}' for i in range(self.ensemble_length)])
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=0.01, weight_decay=5e-4)

            criterion = torch.nn.MSELoss()
            self.model.train()
            for epoch in range(epochs):
                for i, data in enumerate(loader):
                    data.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(data)
                    out = torch.stack(out, dim=0).sum(dim=0)[self.model.batch_mask[:len(getattr(data, f'inp_0'))]]
                    # loss = 0
                    # for j in range(self.ensemble_length):
                    #     loss += criterion(out[j], getattr(data, f'inp_{j}'))
                    loss = criterion(out, getattr(data, f'inp_0')[self.model.batch_mask[:len(getattr(data, f'inp_0'))]])
                    loss.backward()
                    optimizer.step()
            self.model.eval()

    @override
    def retraining(self, training_set):
        assert len(training_set.shape) == 3
        self.train(training_set, epochs=1)

    @override
    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        self.current_predictions, self.current_component_predictions = \
            self.predict(self.current_feature_vectors,
                         return_component_output=True)

    @override
    def predict(self, x, return_component_output=False):
        x_shape = x.shape
        assert len(x_shape) == 3
        output = np.zeros(x_shape)
        data_list = []
        for i in range(x_shape[0]):
            kwargs = {}
            for j in range(self.ensemble_length):
                kwargs[f'inp_{j}'] = torch.from_numpy(
                    np.concatenate([x[i].astype(np.float32), np.zeros((self.nodes_between_length + self.window_length, x[i].shape[1]), dtype=np.float32)]))
                kwargs[f'edge_index_{j}'] = self.edge_indices_list[j]
            data_list.append(EnsembleGNNData(
                ensemble_length=self.ensemble_length, **kwargs))

        pc = torch.unsqueeze(torch.unsqueeze(
            self._performance_weights(), -1), -1)
        if return_component_output:
            component_output = np.zeros(
                (x_shape[0], self.ensemble_length, *x_shape[1:]))
            for i, data in enumerate(data_list):
                out_stacked = torch.stack(self.model(data))
                out_ensemble = torch.sum(pc * out_stacked, dim=0)
                component_output[i] = out_stacked[:, self.model.batch_mask[:len(getattr(data, f'inp_0'))]].detach().numpy()
                output[i] = out_stacked.sum(dim=0)[self.model.batch_mask[:len(getattr(data, f'inp_0'))]].detach().numpy()
            return output, component_output
        else:
            for i, data in enumerate(data_list):
                out_stacked = torch.stack(self.model(data))
                out_ensemble = torch.sum(pc * out_stacked, dim=0)
                output[i] = out_stacked.sum(dim=0)[self.model.batch_mask[:len(getattr(data, f'inp_0'))]].detach().numpy()
            return output
        
    def update_performance_counters_with_anomaly_scores(self, anomaly_score_fn: Callable, anomaly_threshold: float):
        assert len(self.current_component_predictions.shape) == 4
        assert len(self.current_feature_vectors.shape) == 3
        feature_vectors = self.current_feature_vectors[:, np.newaxis]
        individual_scores = calc_nonconformity_scores(feature_vectors, self.current_component_predictions, 'cosine_sim')
        total_scores = calc_nonconformity_scores(self.current_feature_vectors, self.current_predictions, 'cosine_sim')
        
        individual_anomaly_scores = anomaly_score_fn(individual_scores)
        anomaly_scores = anomaly_score_fn(total_scores)
        self.save_component_scores(individual_anomaly_scores, [f'anomaly_scores_{j}' for j in range(self.ensemble_length)])
        for i in range(len(anomaly_scores)):
            for j in range(self.ensemble_length):
                if anomaly_scores[i] > anomaly_threshold:
                    if individual_anomaly_scores[i, j] > anomaly_threshold:
                        self.performance_counters[j] = min(1.0, self.performance_counters[j] + 0.1)
                    else:
                        self.performance_counters[j] = max(-1.0, self.performance_counters[j] - 0.1)
                else:
                    if individual_anomaly_scores[i, j] > anomaly_threshold:
                        self.performance_counters[j] = max(-1.0, self.performance_counters[j] - 0.1)
                    else:
                        self.performance_counters[j] = min(1.0, self.performance_counters[j] + 0.1)

    def save_component_scores(self, scores, column_keys):
        update_index = self.publisher.publisher.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            entry = pd.DataFrame(scores, columns=column_keys, 
                                 index=[i for i in range(update_index-len(scores)+1, update_index+1)])
            if not os.path.exists(save_path):
                entry.index.name = 'counter'
                entry.to_csv(save_path, mode='w')
            else:
                entry.to_csv(save_path, mode='a', header=False)

    @override
    def factory_copy(self):
        return EnsembleGNNWrapper(
            publisher=self.publisher,
            subscribers=copy(self.subscribers),
            model_id=self.model_id,
            model_type=self.model_type,
            window_length=self.window_length,
            ensemble_length=self.ensemble_length,
            model=deepcopy(self.model),
            debug=self.debug)

    @override
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        np.savetxt(f'{save_path.rstrip(".h5")}_performance_counters.csv', self.performance_counters, delimiter=',')


class EnsembleGNN(torch.nn.Module):
    def __init__(self, ensemble_length, window_length, num_node_features, inp_length=1000, batch_size=32):
        super().__init__()
        self.ensemble_length = ensemble_length
        self.window_length = window_length
        self.inp_length = inp_length
        self.num_node_features = num_node_features
        self.batch_size = batch_size
        self.batch_mask = torch.zeros((batch_size * inp_length))
        for j in range(batch_size):
            self.batch_mask[j * inp_length + (inp_length - window_length) : (j+1) * inp_length] = 1
        self.batch_mask = self.batch_mask.bool()
        for i in range(ensemble_length):
            setattr(self, f"conv_{i}_1", GCNConv(num_node_features, 16))
            setattr(self, f"conv_{i}_2", GCNConv(16, 8))
            setattr(self, f"conv_{i}_3", GCNConv(8, 4))
            for j in range(20):
                setattr(self, f"conv_{i}_{4+j}", GCNConv(4, 4))
            setattr(self, f"conv_{i}_24", GCNConv(4, 8))
            setattr(self, f"conv_{i}_25", GCNConv(8, 16))
            setattr(self, f"conv_{i}_26", GCNConv(16, num_node_features))

    def forward(self, data):
        out = []
        for i in range(self.ensemble_length):
            x, edge_index = getattr(data, f'inp_{i}'), getattr(
                data, f'edge_index_{i}')

            for j in range(1, 26):
                x = getattr(self, f"conv_{i}_{j}")(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            out.append(getattr(self, f"conv_{i}_26")(x, edge_index))

        return out


class EnsembleGNNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        for i in range(self.ensemble_length):
            if key == f'edge_index_{i}':
                return getattr(self, f'inp_{i}').size(0)
        return super().__inc__(key, value, *args, **kwargs)
