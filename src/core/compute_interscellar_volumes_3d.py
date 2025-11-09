# Import

import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from scipy.ndimage import label, distance_transform_edt
import sqlite3
import os
import pickle
import math
import zarr

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Data loading

def load_neighbor_pairs_from_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading neighbor pairs from: {csv_path}")
    neighbor_df = pd.read_csv(csv_path)
    
    if 'cell_a_id' in neighbor_df.columns and 'cell_b_id' in neighbor_df.columns:
        pass
    elif 'cell_id_a' in neighbor_df.columns and 'cell_id_b' in neighbor_df.columns:
        neighbor_df = neighbor_df.rename(columns={
            'cell_id_a': 'cell_a_id',
            'cell_id_b': 'cell_b_id'
        })
    else:
        raise ValueError(f"Missing required columns. Found: {list(neighbor_df.columns)}")
    
    if 'cell_a_type' in neighbor_df.columns and 'cell_b_type' in neighbor_df.columns:
        pass
    elif 'cell_type_a' in neighbor_df.columns and 'cell_type_b' in neighbor_df.columns:
        neighbor_df = neighbor_df.rename(columns={
            'cell_type_a': 'cell_a_type',
            'cell_type_b': 'cell_b_type'
        })
    else:
        raise ValueError(f"Missing cell type columns. Found: {list(neighbor_df.columns)}")
    
    print(f"Loaded {len(neighbor_df)} neighbor pairs")
    return neighbor_df

def load_halo_bboxes_from_pickle(pickle_path: str) -> Dict[int, Tuple[slice, slice, slice]]:
    print(f"Loading halo bounding boxes from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Halo bounding boxes pickle file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        bbox_data = pickle.load(f)
    
    if 'all_bboxes_with_halo' in bbox_data:
        halo_bboxes = bbox_data['all_bboxes_with_halo']
    elif isinstance(bbox_data, dict):
        halo_bboxes = bbox_data
    else:
        raise ValueError("Invalid format in halo bounding boxes pickle file")
    
    print(f"Loaded halo bounding boxes for {len(halo_bboxes)} cells")
    return halo_bboxes

def load_global_surface_from_pickle(pickle_path: str) -> np.ndarray:
    print(f"Loading global surface from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Global surface pickle file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        surface_data = pickle.load(f)
    
    if 'global_surface' in surface_data:
        global_surface = surface_data['global_surface']
    elif isinstance(surface_data, np.ndarray):
        global_surface = surface_data
    else:
        raise ValueError("Invalid format in global surface pickle file")
    
    print(f"Loaded global surface with {global_surface.sum()} surface voxels")
    return global_surface

