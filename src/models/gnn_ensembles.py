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
                 model_type: str, window_length: int, number_of_channels: int, ensemble_length: int, model=None, save_paths=None, batch_size=32, debug=False) -> None:
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.dataset_train = None
        self.window_length = window_length
        self.num_latent_nodes = 20
        self.num_latent_node_features = 10
        self.number_of_channels = number_of_channels
        self.nodes_between_length = 100
        self.total_length = self.window_length + self.nodes_between_length + self.num_latent_nodes
        self.ensemble_length = ensemble_length
        self.edge_indices_list = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = model.to(self.device)
        self.encoder = EnsembleGNN(
            input_length=self.window_length,
            between_length=self.nodes_between_length,
            output_length=self.num_latent_nodes,
            ensemble_length=ensemble_length,
            num_node_features_input=number_of_channels,
            num_node_features_output=self.num_latent_node_features,
            edge_idx_base_key='enc_edge_index',
            batch_size=batch_size,
        )
        self.decoder1 = EnsembleGNN(
            input_length=self.num_latent_nodes,
            between_length=self.nodes_between_length,
            output_length=self.window_length,
            ensemble_length=ensemble_length,
            num_node_features_input=self.num_latent_node_features,
            num_node_features_output=number_of_channels,
            edge_idx_base_key='dec1_edge_index',
            batch_size=batch_size,
        )
        self.decoder2 = EnsembleGNN(
            input_length=self.num_latent_nodes,
            between_length=self.nodes_between_length,
            output_length=self.window_length,
            ensemble_length=ensemble_length,
            num_node_features_input=self.num_latent_node_features,
            num_node_features_output=number_of_channels,
            edge_idx_base_key='dec2_edge_index',
            batch_size=batch_size,
        )
        self.performance_counters = np.zeros((ensemble_length))
        self.save_paths = save_paths
        self.batch_size = batch_size
        self.debug = debug
        
        self.epoch = 1

        self.encoder_edge_indices_list = []
        self.decoder1_edge_indices_list = []
        self.decoder2_edge_indices_list = []
        self._sample_edge_indices_by_routes(
            edge_indices_list=self.encoder_edge_indices_list,
            input_length=self.window_length,
            between_length=self.nodes_between_length,
            output_length=self.num_latent_nodes,
        )
        self._sample_edge_indices_by_routes(
            edge_indices_list=self.decoder1_edge_indices_list,
            input_length=self.num_latent_nodes,
            between_length=self.nodes_between_length,
            output_length=self.window_length,
        )
        self._sample_edge_indices_by_routes(
            edge_indices_list=self.decoder2_edge_indices_list,
            input_length=self.num_latent_nodes,
            between_length=self.nodes_between_length,
            output_length=self.window_length,
        )

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
            

    def _sample_edge_indices_by_routes(self, edge_indices_list, input_length, output_length, between_length):
        routes_per_component = 10
        component_start_target_len = 10
        component_end_target_len = 2
        max_route_len = 20
        start_nodes = np.arange(0, input_length)
        end_nodes = np.arange(input_length + between_length, input_length + between_length + output_length)
        for comp_idx in range(self.ensemble_length):
            comp_start_idx = comp_idx * int(input_length / self.ensemble_length)
            comp_end_idx = comp_idx * int(output_length / self.ensemble_length)
            comp_start_nodes = start_nodes[comp_start_idx : comp_start_idx + component_start_target_len]
            comp_end_nodes = end_nodes[comp_end_idx : comp_end_idx + component_end_target_len]
            edges = []
            for i in range(routes_per_component):
                comp_start_nodes_idx = np.random.randint(0, len(comp_start_nodes))
                comp_end_nodes_idx = np.random.randint(0, len(comp_end_nodes))
                j = comp_start_nodes[comp_start_nodes_idx]
                k = 0
                comp_offset = comp_idx * int(between_length / self.ensemble_length)
                j += comp_offset
                while j < comp_end_nodes[comp_end_nodes_idx]:
                    # draw = np.random.random()
                    # next_node_offset = int(1 + -np.ceil(np.log2(draw)))
                    next_node_offset = np.random.randint(1, 10)
                    edges.append((j, min(j + next_node_offset, comp_end_nodes[comp_end_nodes_idx])))
                    j += next_node_offset
                    k += 1
                    if k == max_route_len and j < comp_end_nodes[comp_end_nodes_idx]:
                        edges.append((j, comp_end_nodes[comp_end_nodes_idx]))
                        break
            edge_indices_list.append(torch.tensor([
                [edges[j][0] for j in range(len(edges))],
                [edges[j][1] for j in range(len(edges))]
            ]))
            
    def _sample_edge_indices_by_noncausal_routes(self, edge_indices_list, input_length, output_length, between_length):
        routes_per_component = 10
        component_start_target_len = 10
        component_end_target_len = 2
        max_route_len = 20
        start_nodes = np.arange(0, input_length)
        end_nodes = np.arange(input_length + between_length, input_length + between_length + output_length)
        for comp_idx in range(self.ensemble_length):
            comp_start_idx = comp_idx * int(input_length / self.ensemble_length)
            comp_end_idx = comp_idx * int(output_length / self.ensemble_length)
            comp_start_nodes = start_nodes[comp_start_idx : comp_start_idx + component_start_target_len]
            comp_end_nodes = end_nodes[comp_end_idx : comp_end_idx + component_end_target_len]
            edges = []
            for i in range(routes_per_component):
                comp_start_nodes_idx = np.random.randint(0, len(comp_start_nodes))
                comp_end_nodes_idx = np.random.randint(0, len(comp_end_nodes))
                j = comp_start_nodes[comp_start_nodes_idx]
                for k in range(max_route_len-1):
                    next_node = np.random.randint(0, comp_end_nodes[comp_end_nodes_idx]-1)
                    edges.append((j, next_node))
                    j = next_node
                edges.append((j, comp_end_nodes[comp_end_nodes_idx]))
            edge_indices_list.append(torch.tensor([
                [edges[j][0] for j in range(len(edges))],
                [edges[j][1] for j in range(len(edges))]
            ]))
    
    def _calc_number_of_routes(self, dist):
        return 1 + sum([2**(dist - 1 - j) for j in range(dist-1, 0, -1)])

    def _performance_weights(self):
        return F.softmax(torch.tensor(self.performance_counters), dim=0)

    @override
    def train(self, x: np.ndarray, epochs):
        data_list = []
        for i in range(len(x)):
            kwargs = {}
            for j in range(self.ensemble_length):
                kwargs[f'inp_{j}'] = torch.from_numpy(
                    np.concatenate([x[i].astype(np.float32), np.zeros((self.nodes_between_length + self.num_latent_nodes, x[i].shape[1]), dtype=np.float32)]))
                kwargs[f'enc_edge_index_{j}'] = self.encoder_edge_indices_list[j]
                kwargs[f'dec1_edge_index_{j}'] = self.decoder1_edge_indices_list[j]
                kwargs[f'dec2_edge_index_{j}'] = self.decoder2_edge_indices_list[j]
            data_list.append(EnsembleGNNData(
                ensemble_length=self.ensemble_length, **kwargs))

        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True, follow_batch=[
                            f'inp_{i}' for i in range(self.ensemble_length)])
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder1.parameters()) + list(self.decoder2.parameters()), lr=0.01, weight_decay=5e-4)

        criterion = torch.nn.MSELoss()
        self.encoder.train()
        self.decoder1.train()
        self.decoder2.train()
        for epoch in range(epochs):
            for i, data in enumerate(loader):
                data.to(self.device)
                optimizer.zero_grad()
                tmp_mask = torch.zeros((data.inp_0.size(0)))
                for j in range(data.inp_0.size(0) // self.total_length):
                    tmp_mask[j * self.total_length : j * self.total_length + self.window_length] = 1
                input_batch_unpadded = data.inp_0[tmp_mask.bool()]
                enc_out = self.encoder(data)
                enc_out = torch.stack(enc_out, dim=0).sum(dim=0)[self.encoder.batch_mask[:enc_out[0].size(0)]]
                batch_size = enc_out.size(0) // self.num_latent_nodes
                temp = torch.zeros((batch_size * self.total_length, self.num_latent_node_features))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.num_latent_nodes] = enc_out[j * self.num_latent_nodes : (j+1) * self.num_latent_nodes]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                dec1_out = self.decoder1(data)
                dec1_out = torch.stack(dec1_out, dim=0).sum(dim=0)[self.decoder1.batch_mask[:dec1_out[0].size(0)]]
                dec1_loss = criterion(dec1_out, input_batch_unpadded)
                
                dec2_out = self.decoder2(data)
                dec2_out = torch.stack(dec2_out, dim=0).sum(dim=0)[self.decoder2.batch_mask[:dec2_out[0].size(0)]]
                dec2_loss = criterion(dec2_out, input_batch_unpadded)
                
                temp = torch.zeros((batch_size * self.total_length, self.number_of_channels))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.window_length] = dec1_out[j * self.window_length : (j+1) * self.window_length]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                out_3 = self.encoder(data)
                out_3 = torch.stack(out_3, dim=0).sum(dim=0)[self.encoder.batch_mask[:out_3[0].size(0)]]
                temp = torch.zeros((batch_size * self.total_length, self.num_latent_node_features))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.num_latent_nodes] = enc_out[j * self.num_latent_nodes : (j+1) * self.num_latent_nodes]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                out_3 = self.decoder2(data)
                out_3 = torch.stack(out_3, dim=0).sum(dim=0)[self.decoder2.batch_mask[:out_3[0].size(0)]]
                out_3_loss = criterion(out_3, input_batch_unpadded)
                
                loss1 = (1 / self.epoch) * dec1_loss + (1 - 1 / self.epoch) * out_3_loss
                loss2 = (1 / self.epoch) * dec2_loss - (1 - 1 / self.epoch) * out_3_loss
                loss1.backward()
                loss2.backward()
                optimizer.step()
            self.epoch += 1
            print(f'Finished epoch {epoch}')
        self.encoder.eval()
        self.decoder1.eval()
        self.decoder2.eval()

    @override
    def retraining(self, training_set):
        assert len(training_set.shape) == 3
        self.train(training_set, epochs=1)

    @override
    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        self.current_predictions = \
            self.predict(self.current_feature_vectors,
                         return_component_output=False)

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
                    np.concatenate([x[i].astype(np.float32), np.zeros((self.nodes_between_length + self.num_latent_nodes, x[i].shape[1]), dtype=np.float32)]))
                kwargs[f'enc_edge_index_{j}'] = self.encoder_edge_indices_list[j]
                kwargs[f'dec1_edge_index_{j}'] = self.decoder1_edge_indices_list[j]
                kwargs[f'dec2_edge_index_{j}'] = self.decoder2_edge_indices_list[j]
            data_list.append(EnsembleGNNData(
                ensemble_length=self.ensemble_length, **kwargs))

        if return_component_output:
            # pc = torch.unsqueeze(torch.unsqueeze(
            #     self._performance_weights(), -1), -1)
            # component_output = np.zeros(
            #     (x_shape[0], self.ensemble_length, *x_shape[1:]))
            # for i, data in enumerate(data_list):
            #     out_stacked = torch.stack(self.model(data))
            #     out_ensemble = torch.sum(pc * out_stacked, dim=0)
            #     component_output[i] = out_stacked[:, self.model.batch_mask[:len(getattr(data, f'inp_0'))]].detach().numpy()
            #     output[i] = out_stacked.sum(dim=0)[self.model.batch_mask[:len(getattr(data, f'inp_0'))]].detach().numpy()
            # return output, component_output
            return None
        else:
            for data_idx, data in enumerate(data_list):
                data.to(self.device)
                enc_out = self.encoder(data)
                enc_out = torch.stack(enc_out, dim=0).sum(dim=0)[self.encoder.batch_mask[:enc_out[0].size(0)]]
                batch_size = enc_out.size(0) // self.num_latent_nodes
                temp = torch.zeros((batch_size * self.total_length, self.num_latent_node_features))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.num_latent_nodes] = enc_out[j * self.num_latent_nodes : (j+1) * self.num_latent_nodes]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                dec1_out = self.decoder1(data)
                dec1_out = torch.stack(dec1_out, dim=0).sum(dim=0)[self.decoder1.batch_mask[:dec1_out[0].size(0)]]
                dec2_out = self.decoder2(data)
                dec2_out = torch.stack(dec2_out, dim=0).sum(dim=0)[self.decoder2.batch_mask[:dec2_out[0].size(0)]]
                
                temp = torch.zeros((batch_size * self.total_length, self.number_of_channels))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.window_length] = dec1_out[j * self.window_length : (j+1) * self.window_length]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                out_3 = self.encoder(data)
                out_3 = torch.stack(out_3, dim=0).sum(dim=0)[self.encoder.batch_mask[:out_3[0].size(0)]]
                temp = torch.zeros((batch_size * self.total_length, self.num_latent_node_features))
                for j in range(batch_size):
                    temp[j * self.total_length : j * self.total_length + self.num_latent_nodes] = enc_out[j * self.num_latent_nodes : (j+1) * self.num_latent_nodes]
                for i in range(self.ensemble_length):
                    setattr(data, f'inp_{i}', temp)
                out_3 = self.decoder2(data)
                out_3 = torch.stack(out_3, dim=0).sum(dim=0)[self.decoder2.batch_mask[:out_3[0].size(0)]]
                output[data_idx] = out_3.detach().numpy()
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
        torch.save(self.encoder.state_dict(), f'{save_path.rstrip(".h5")}_encoder.h5')
        torch.save(self.decoder1.state_dict(), f'{save_path.rstrip(".h5")}_decoder1.h5')
        torch.save(self.decoder2.state_dict(), f'{save_path.rstrip(".h5")}_decoder2.h5')
        # np.savetxt(f'{save_path.rstrip(".h5")}_performance_counters.csv', self.performance_counters, delimiter=',')


class EnsembleGNN(torch.nn.Module):
    def __init__(self, ensemble_length, num_node_features_input, num_node_features_output, input_length, output_length, between_length, edge_idx_base_key='edge_index', num_conv_layers=26, batch_size=32):
        super().__init__()
        self.ensemble_length = ensemble_length
        self.input_length = input_length
        self.output_length = output_length
        self.between_length = between_length
        self.total_length = input_length + output_length + between_length
        self.num_node_features = num_node_features_input
        self.num_latent_node_features = num_node_features_output
        self.num_conv_layers = num_conv_layers
        self.edge_idx_base_key = edge_idx_base_key
        self.batch_size = batch_size
        self.batch_mask = torch.zeros((batch_size * self.total_length))
        for j in range(batch_size):
            self.batch_mask[j * self.total_length + (self.total_length - self.output_length) : (j+1) * self.total_length] = 1
        self.batch_mask = self.batch_mask.bool()
        
        node_feature_lengths = np.ceil(np.linspace(num_node_features_input, num_node_features_output, num_conv_layers+1))
        for i in range(ensemble_length):
            for j in range(1, self.num_conv_layers+1):
                setattr(self, f"conv_{i}_{j}", GCNConv(int(node_feature_lengths[j-1]), int(node_feature_lengths[j])))

    def forward(self, data):
        out = []
        for i in range(1, self.ensemble_length):
            x, edge_index = getattr(data, f'inp_{i}'), getattr(
                data, f'{self.edge_idx_base_key}_{i}')

            for j in range(1, self.num_conv_layers):
                x = getattr(self, f"conv_{i}_{j}")(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            out.append(getattr(self, f"conv_{i}_{self.num_conv_layers}")(x, edge_index))

        return out


class EnsembleGNNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        for i in range(self.ensemble_length):
            if f'edge_index_{i}' in key:
                return getattr(self, f'inp_{i}').size(0)
        return super().__inc__(key, value, *args, **kwargs)

