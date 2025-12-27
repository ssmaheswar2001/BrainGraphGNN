"""
PyTorch Geometric dataset classes for brain connectivity graphs
"""

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler

from ..utils.features import extract_rich_node_features, adaptive_threshold


class MultiConnectivityDataset(InMemoryDataset):
    """
    Dataset with multiple connectivity measures as edge features
    """
    
    def __init__(self, root, time_series, labels, threshold=0.3,
                 transform=None, pre_transform=None):
        self.time_series = time_series
        self.labels = labels
        self.threshold = threshold
        self.n_rois = time_series[0].shape[1]
        self.n_participants = len(time_series)
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['multi_connectivity_graphs.pt']
    
    def download(self):
        pass
    
    def process(self):
        """
        Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.
        """
        
        corr_measure = ConnectivityMeasure(kind='correlation')
        pcorr_measure = ConnectivityMeasure(kind='partial correlation')
        tan_measure = ConnectivityMeasure(kind='tangent')
        
        print('Computing correlation....')
        corr_matrices = corr_measure.fit_transform(self.time_series)
        print('Computing partial correlation....')
        pcorr_matrices = pcorr_measure.fit_transform(self.time_series)
        print('Computing tangent....')
        tan_matrices = tan_measure.fit_transform(self.time_series)
        
        graphs = []
        
        for idx in range(self.n_participants):
            ts = self.time_series[idx]
            
            # Node features
            x = torch.tensor(np.column_stack([
                np.mean(ts, axis=0),
                np.std(ts, axis=0),
            ]), dtype=torch.float)
            
            # Build edges using correlation, but include both measures as features
            edge_index = []
            edge_attr = []
            
            for i in range(self.n_rois):
                for j in range(i+1, self.n_rois):
                    if abs(corr_matrices[idx][i, j]) > self.threshold:
                        edge_index.extend([[i, j], [j, i]])
                        
                        # Multiple edge features
                        feat = [
                            corr_matrices[idx][i, j],
                            pcorr_matrices[idx][i, j],
                            abs(corr_matrices[idx][i, j]),
                            1.0 if corr_matrices[idx][i, j] > 0 else -1.0,  # Sign
                        ]
                        edge_attr.extend([feat, feat])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            y = torch.tensor([1 if self.labels[idx] == 'adult' else 0], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=self.n_rois)
            graphs.append(data)
        
        print(f"Created {len(graphs)} graphs")
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved to {self.processed_paths[0]}")
    
    def get_labels(self):
        return torch.cat([self[i].y for i in range(len(self))], dim=0)
    
    def get_statistics(self):
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"Number of graphs: {len(self)}")
        print(f"Number of nodes per graph: {self.n_rois}")
        print(f"Number of node features: {self[0].x.shape[1]}")
        print(f"Number of edge features: {self[0].edge_attr.shape[1] if self[0].edge_attr.numel() > 0 else 0}")
        
        # Edge statistics
        num_edges = [data.edge_index.shape[1] for data in self]
        print(f"Average edges: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
        print(f"Min edges: {np.min(num_edges)}, Max edges: {np.max(num_edges)}")
        
        # Label distribution
        labels = self.get_labels().numpy()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        for label, count in zip(unique, counts):
            label_name = 'Child' if label == 0 else 'Adult'
            print(f"  {label_name}: {count} ({count/len(labels)*100:.1f}%)")
        print(f"{'='*60}\n")


class BrainDataset(InMemoryDataset):
    """
    Dataset with all fixes applied:
    - Rich node features (14 features)
    - Adaptive edge thresholding
    - Feature normalization
    - Better edge features
    """
    
    def __init__(self, root, time_series, labels, threshold_percentile=85,
                 normalize_features=True, transform=None, pre_transform=None):
        self.time_series = time_series
        self.labels = labels
        self.threshold_percentile = threshold_percentile
        self.normalize_features = normalize_features
        self.n_rois = time_series[0].shape[1]
        self.n_participants = len(time_series)
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['brain_graphs.pt']
    
    def download(self):
        pass
    
    def process(self):
        print('\nProcessing dataset...')
        print('Computing connectivity matrices...')
        
        corr_measure = ConnectivityMeasure(kind='correlation')
        pcorr_measure = ConnectivityMeasure(kind='partial correlation')
        
        corr_matrices = corr_measure.fit_transform(self.time_series)
        pcorr_matrices = pcorr_measure.fit_transform(self.time_series)
        
        print('Extracting features...')
        all_node_features = []
        all_edge_features = []
        graph_info = []
        
        # First pass: extract all features
        for idx in range(self.n_participants):
            ts = self.time_series[idx]
            
            # Rich node features
            node_feat = extract_rich_node_features(ts)
            all_node_features.append(node_feat)
            
            # Adaptive threshold
            threshold = adaptive_threshold(
                corr_matrices[idx],
                percentile=self.threshold_percentile
            )
            
            # Build edges
            edge_index = []
            edge_attr = []
            
            for i in range(self.n_rois):
                for j in range(i+1, self.n_rois):
                    if abs(corr_matrices[idx][i, j]) > threshold:
                        edge_index.extend([[i, j], [j, i]])
                        
                        feat = [
                            corr_matrices[idx][i, j],
                            pcorr_matrices[idx][i, j],
                            abs(corr_matrices[idx][i, j]),
                            1.0 if corr_matrices[idx][i, j] > 0 else -1.0,
                        ]
                        edge_attr.extend([feat, feat])
            
            all_edge_features.append(np.array(edge_attr) if edge_attr else np.array([]))
            graph_info.append({
                'edge_index': edge_index,
                'n_edges': len(edge_index)
            })
        
        # Normalize features
        if self.normalize_features:
            print('Normalizing features...')
            all_nodes_stacked = np.vstack(all_node_features)
            node_scaler = StandardScaler()
            node_scaler.fit(all_nodes_stacked)
            
            edge_features_list = [ef for ef in all_edge_features if len(ef) > 0]
            if edge_features_list:
                all_edges_stacked = np.vstack(edge_features_list)
                edge_scaler = StandardScaler()
                edge_scaler.fit(all_edges_stacked)
            else:
                edge_scaler = None
        
        # Create graphs
        print('Creating graph objects...')
        graphs = []
        
        for idx in range(self.n_participants):
            # Node features
            node_feat = all_node_features[idx]
            if self.normalize_features:
                node_feat = node_scaler.transform(node_feat)
            x = torch.tensor(node_feat, dtype=torch.float)
            
            # Edge features
            if graph_info[idx]['n_edges'] > 0:
                edge_index = torch.tensor(
                    graph_info[idx]['edge_index'],
                    dtype=torch.long
                ).t().contiguous()
                
                edge_attr = all_edge_features[idx]
                if self.normalize_features and edge_scaler is not None:
                    edge_attr = edge_scaler.transform(edge_attr)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                edge_index = torch.tensor(
                    [[i, i] for i in range(self.n_rois)],
                    dtype=torch.long
                ).t().contiguous()
                edge_attr = torch.zeros((self.n_rois, 4), dtype=torch.float)
            
            # Label
            y = torch.tensor(
                [1 if self.labels[idx] == 'adult' else 0],
                dtype=torch.long
            )
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=self.n_rois
            )
            graphs.append(data)
        
        print(f'Created {len(graphs)} graphs')
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        print(f'Saved to {self.processed_paths[0]}')
    
    def get_labels(self):
        return torch.cat([self[i].y for i in range(len(self))], dim=0)
    
    def get_statistics(self):
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"Number of graphs: {len(self)}")
        print(f"Number of nodes per graph: {self.n_rois}")
        print(f"Number of node features: {self[0].x.shape[1]}")
        print(f"Number of edge features: {self[0].edge_attr.shape[1]}")
        
        num_edges = [data.edge_index.shape[1] for data in self]
        print(f"Average edges: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
        print(f"Min edges: {np.min(num_edges)}, Max edges: {np.max(num_edges)}")
        
        labels = self.get_labels().numpy()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        for label, count in zip(unique, counts):
            label_name = 'Child' if label == 0 else 'Adult'
            print(f"  {label_name}: {count} ({count/len(labels)*100:.1f}%)")
        print(f"{'='*60}\n")