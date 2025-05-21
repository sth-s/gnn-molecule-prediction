"""
Utility functions for loading and preprocessing the Tox21 dataset.
"""
import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from tqdm import tqdm
import requests
import shutil


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
                 target_cols: list[str] | None = None,
                 cache_file: str = "processed_tox21.pt",
                 download: bool = True,
                 recreate: bool = False,
                 transform=None,
                 pre_transform=None):
        """
        Tox21 dataset for molecular toxicity prediction.
        
        Parameters:
        ── root (str): Root directory where the dataset should be saved
        ── filename (str): Name of the raw file to download
        ── smiles_col (str): Column name in CSV that contains SMILES strings
        ── target_cols (list[str]|None): Target columns to predict. If None, all columns except smiles_col.
        ── transform (callable, optional): Transform to be applied to each data instance
        ── pre_transform (callable, optional): Transform to be applied to dataset before saving
        ── recreate (bool): If True, dataset will be reprocessed even if processed files exist
        """
        self.filename   = filename
        self._target_cols = target_cols
        self.cache_file = cache_file
        self.download   = download
        self.recreate   = recreate
        # if recreate=True, remove previously generated cache
        if self.recreate and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [self.cache_file]

    def download(self):
        if not self.download:
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
            self._actual_target_cols = [c for c in df.columns if c != self.smiles_col]
        else:
            self._actual_target_cols = self._target_cols
        
        # Process molecules
        data_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                print(f"Failed SMILES: {smiles} (idx={idx})")
                continue
            
            # Node features
            atom_feats = []
            for atom in mol.GetAtoms():
                atom_feats.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetTotalNumHs(),
                    int(atom.GetIsAromatic()),
                ])
            x = torch.tensor(atom_feats, dtype=torch.float)
            
            # Edges + edge features
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index += [[i, j], [j, i]]
                # example of bond feature: type (1-single, 2-double...)
                bond_type = bond.GetBondTypeAsDouble()
                edge_attr += [[bond_type], [bond_type]]
            
            if len(edge_index) > 0:  # Check that there is at least one edge
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                # Create empty tensors of appropriate dimensions
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            
            # Labels
            labels = [float(row[c]) for c in self._actual_target_cols]
            y = torch.tensor(labels, dtype=torch.float)
            
            # Create mask for NaN values and replace NaNs with zeros
            y_mask = ~torch.isnan(y)
            y = torch.nan_to_num(y, nan=0.0)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_mask=y_mask)
            
            # Apply pre-transform if defined
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"Dataset processed and saved to: {self.processed_paths[0]}")
        print(f"Total graphs: {len(data_list)}")
        

def load_tox21(
    root: str = "data",
    filename: str = "tox21.csv",
    smiles_col: str = "smiles",
    target_cols: list[str] | None = None,
    cache_file: str = "data.pt",
    recreate: bool = False,
    download: bool = True,
) -> Tox21Dataset:
    """
    Loads and caches Tox21 as a PyG InMemoryDataset.

    Parameters:
    ── root (str): path to the dataset directory (will contain raw/ and processed/ subdirectories).
    ── filename (str): name of the CSV file in the raw/ directory.
    ── smiles_col (str): name of the column with SMILES.
    ── target_cols (list[str]|None): list of target columns. 
         If None — all columns except smiles_col.
    ── cache_file (str): name of the cache file (torch.save) in the processed/ directory.
    ── recreate (bool): if True — ignore cache and recreate.
    ── download (bool): if True — automatically download the dataset if it doesn't exist.

    Returns:
    ── dataset (Tox21Dataset): contains Data objects with fields:
         • x: FloatTensor[num_atoms, num_node_features]
         • edge_index: LongTensor[2, num_edges]
         • edge_attr (optional): FloatTensor[num_edges, num_edge_features]
         • y: FloatTensor[num_tasks]
    """
    # Simply create and return a Tox21Dataset instance
    # The dataset will handle downloading and processing automatically
    return Tox21Dataset(root, filename, target_cols, cache_file, download, recreate)