"""
Utility functions for loading and preprocessing the Tox21 dataset.
"""
import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import requests
import shutil
import deepchem as dc
from deepchem.feat.graph_data import GraphData

# Setup for safe weights-only loading in PyTorch 2.6+
try:
    from torch import serialization
    # Import PyG classes for serialization
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    
    try:
        from torch_geometric.data.storage import NodeStorage, EdgeStorage, GraphStorage
        pyg_storage_classes = [GlobalStorage, NodeStorage, EdgeStorage, GraphStorage]
    except ImportError:
        pyg_storage_classes = [GlobalStorage]
    
    # Register classes for safe loading
    safe_classes = [DataEdgeAttr, DataTensorAttr] + pyg_storage_classes
    serialization.add_safe_globals(safe_classes)
    print(f"Registered PyG classes for safe weights_only loading")
except (ImportError, AttributeError) as e:
    print(f"Note: Using default PyTorch loading behavior")


def download_tox21(root: str, filename: str) -> str:
    """
    Downloads the Tox21 dataset if it doesn't exist.
    
    Parameters:
    ── root (str): path to the directory where the file will be downloaded
    ── filename (str): name of the file to save
    
    Returns:
    ── path (str): full path to the downloaded file
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    
    # Ensure we're saving to the raw directory
    raw_dir = os.path.join(root, 'raw')
    full_path = os.path.join(raw_dir, filename)
    
    # Check and create directory if needed
    os.makedirs(raw_dir, exist_ok=True)
    
    # If the file already exists, just return the path
    if os.path.exists(full_path):
        print(f"File already exists: {full_path}")
        return full_path
    
    # Download file
    print(f"Downloading Tox21 dataset from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check that the request was successful
    
    # First save the gzip file
    gz_path = full_path + ".gz"
    with open(gz_path, "wb") as f:
        shutil.copyfileobj(response.raw, f)
    
    # Extract gzip to CSV
    print(f"Extracting to {full_path}...")
    import gzip
    with gzip.open(gz_path, 'rb') as f_in:
        with open(full_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the archive
    os.remove(gz_path)
    
    print(f"Dataset successfully downloaded and extracted to: {full_path}")
    return full_path


# Creating InMemoryDataset
class Tox21Dataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 filename: str = "tox21.csv",
                 smiles_col: str = "smiles",
                 mol_id_col: str = "mol_id",
                 target_cols: list[str] | None = None,
                 cache_file: str = "processed_tox21.pt",                
                 auto_download: bool = True,
                 recreate: bool = False,
                 transform=None,
                 pre_transform=None):
        """
        Tox21 dataset for molecular toxicity prediction.
        
        Parameters:
        ── root (str): Root directory where the dataset should be saved
        ── filename (str): Name of the raw file to download
        ── smiles_col (str): Column name in CSV that contains SMILES strings
        ── mol_id_col (str): Column name in CSV that contains molecule IDs
        ── target_cols (list[str]|None): Target columns to predict. If None, all columns except smiles_col and mol_id_col.
        ── cache_file (str): Name of the cache file for processed data
        ── download (bool): If True, automatically download the dataset if it doesn't exist
        ── recreate (bool): If True, dataset will be reprocessed even if processed files exist
        ── transform (callable, optional): Transform to be applied to each data instance
        ── pre_transform (callable, optional): Transform to be applied to dataset before saving
        """
        self.filename   = filename
        self.smiles_col = smiles_col
        self.mol_id_col = mol_id_col
        self._target_cols = target_cols
        self.cache_file = cache_file
        self.auto_download = auto_download
        self.recreate   = recreate
        
        _processed_dir = os.path.join(root, 'processed')
        _processed_file_path = os.path.join(_processed_dir, self.cache_file)

        if self.recreate and os.path.exists(_processed_file_path):
            print(f"Recreating dataset: removing existing cache file")
            os.makedirs(_processed_dir, exist_ok=True) 
            os.remove(_processed_file_path)
        
        # Call the parent constructor which handles dataset setup
        super().__init__(root, transform, pre_transform)
        
        # Explicitly load the processed data if file exists
        # This ensures compatibility with PyTorch 2.6+ where weights_only=True by default
        if os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [self.cache_file]

    def download(self):
        if not self.auto_download:
            raise RuntimeError("download=False, raw file missing")
        download_tox21(self.root, self.filename)
    
    @property
    def target_cols(self):
        if not hasattr(self, '_actual_target_cols'):
            # This will be set during processing
            raise AttributeError("Target columns are only available after processing")
        return self._actual_target_cols
    
    @property
    def num_classes(self):
        return len(self.target_cols)
    
    def process(self):
        # Read CSV file
        csv_path = os.path.join(self.raw_dir, self.filename)
        print(f"Loading dataset from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Determine target columns
        if self._target_cols is None:
            # Exclude smiles_col AND mol_id_col if target_cols are not specified
            self._actual_target_cols = [
                c for c in df.columns if c not in [self.smiles_col, self.mol_id_col]
            ]
        else:
            self._actual_target_cols = self._target_cols
        
        # Initialize molecule featurizer from DeepChem
        featurizer = dc.feat.GraphConvFeaturizer(use_edges=True)
        
        # Process molecules
        data_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            smiles = row[self.smiles_col]
            mol_id = str(row[self.mol_id_col])  # Ensure mol_id is string
            
            # Use DeepChem to featurize the molecule
            try:
                # Featurize the molecule using DeepChem
                mol_graphs = featurizer.featurize([smiles])
                if not mol_graphs or mol_graphs[0] is None:
                    print(f"Failed SMILES: {smiles} (idx={idx}, mol_id={mol_id})")
                    continue
                
                mol_graph = mol_graphs[0]
                
                # Extract node features directly from GraphData
                x = torch.tensor(mol_graph.node_features, dtype=torch.float)
                
                # Extract edge information directly from GraphData
                edge_index = torch.tensor(mol_graph.edge_index.T, dtype=torch.long)
                edge_attr = torch.tensor(mol_graph.edge_features, dtype=torch.float)
                
                if edge_index.shape[1] == 0:  # Check that there is at least one edge
                    # Create empty tensors of appropriate dimensions
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros((0, 1), dtype=torch.float)
                
                # Labels
                labels = [float(row[c]) if pd.notna(row[c]) else float('nan') for c in self._actual_target_cols]
                y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)  # Ensure y is [1, num_tasks]
                
                # Create mask for NaN values and replace NaNs with zeros
                y_mask = ~torch.isnan(y)
                y = torch.nan_to_num(y, nan=0.0)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_mask=y_mask, mol_id=mol_id)
                
                # Apply pre-transform if defined
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing SMILES {smiles} (idx={idx}, mol_id={mol_id}): {e}")
                continue
        
        # Handle empty dataset case
        if not data_list:
            print("Warning: No molecules were successfully processed. Creating empty dataset.")
            # Dynamically determine the number of node features from the first successfully processed graph
            # or use a default value if no graphs were successfully processed
            num_node_features = mol_graph.node_features.shape[1] if mol_graph else 75
            
            # The dimensionality of edge features is typically 1 but may vary depending on the featurizer
            num_edge_features = mol_graph.edge_features.shape[1] if mol_graph and mol_graph.edge_features.shape[0] > 0 else 1
            
            dummy_data = Data(x=torch.empty((0, num_node_features), dtype=torch.float),
                            edge_index=torch.empty((2, 0), dtype=torch.long),
                            edge_attr=torch.empty((0, num_edge_features), dtype=torch.float),
                            y=torch.empty((1, len(self._actual_target_cols)), dtype=torch.float),
                            y_mask=torch.empty((1, len(self._actual_target_cols)), dtype=torch.bool),
                            mol_id="dummy_molecule")
            data_list.append(dummy_data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"Dataset processed and saved to: {self.processed_paths[0]}")
        print(f"Total graphs: {len(data_list)}")
        

def load_tox21(
    root: str = "data",
    filename: str = "tox21.csv",
    smiles_col: str = "smiles",
    mol_id_col: str = "mol_id",  # New parameter
    target_cols: list[str] | None = None,
    cache_file: str = "data.pt",
    recreate: bool = False,
    auto_download: bool = True,
) -> Tox21Dataset:
    """
    Loads and caches Tox21 as a PyG InMemoryDataset.

    Parameters:
    ── root (str): path to the dataset directory (will contain raw/ and processed/ subdirectories).
    ── filename (str): name of the CSV file in the raw/ directory.
    ── smiles_col (str): name of the column with SMILES.
    ── mol_id_col (str): name of the column with molecule IDs.
    ── target_cols (list[str]|None): list of target columns. 
         If None — all columns except smiles_col and mol_id_col.
    ── cache_file (str): name of the cache file (torch.save) in the processed/ directory.
    ── recreate (bool): if True — ignore cache and recreate.
    ── auto_download (bool): if True — automatically download the dataset if it doesn't exist.

    Returns:
    ── dataset (Tox21Dataset): contains Data objects with fields:
         • x: FloatTensor[num_atoms, num_node_features]
         • edge_index: LongTensor[2, num_edges]
         • edge_attr (optional): FloatTensor[num_edges, num_edge_features]
         • y: FloatTensor[num_tasks]
         • mol_id: (str) molecule identifier
    """
    # Simply create and return a Tox21Dataset instance
    # The dataset will handle downloading and processing automatically
    return Tox21Dataset(
        root=root, 
        filename=filename,
        smiles_col=smiles_col,
        mol_id_col=mol_id_col,  # Pass mol_id_col
        target_cols=target_cols, 
        cache_file=cache_file, 
        auto_download=auto_download, 
        recreate=recreate
    )