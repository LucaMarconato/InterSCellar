# Import

import numpy as np
import pandas as pd
from itertools import product
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from skimage.measure import regionprops
import skimage.measure
sk_label = skimage.measure.label
from scipy.ndimage import label, generate_binary_structure, binary_erosion, distance_transform_edt, find_objects
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import cv2
import csv
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

## Cell surface precomputation

def get_bounding_boxes_3d(mask_3d: np.ndarray, unique_ids: set) -> dict:
    z, y, x = np.nonzero(mask_3d)
    cell_ids = mask_3d[z, y, x]
    df = pd.DataFrame({'cell_id': cell_ids, 'z': z, 'y': y, 'x': x})
    bbox = {}
    grouped = df.groupby('cell_id')
    for cell_id, group in grouped:
        if cell_id == 0:
            continue
        minz, maxz = group['z'].min(), group['z'].max() + 1
        miny, maxy = group['y'].min(), group['y'].max() + 1
        minx, maxx = group['x'].min(), group['x'].max() + 1
        bbox[cell_id] = (slice(minz, maxz), slice(miny, maxy), slice(minx, maxx))
    return bbox

def compute_bounding_box_with_halo(
    surface_a: np.ndarray,
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[slice, slice, slice]:
    z_coords, y_coords, x_coords = np.where(surface_a)
    
    if len(z_coords) == 0:
        return None
    
    min_z, max_z = z_coords.min(), z_coords.max() + 1
    min_y, max_y = y_coords.min(), y_coords.max() + 1
    min_x, max_x = x_coords.min(), x_coords.max() + 1
    
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    min_z_pad = max(0, min_z - pad_z)
    max_z_pad = max_z + pad_z + 1
    min_y_pad = max(0, min_y - pad_y)
    max_y_pad = max_y + pad_y + 1
    min_x_pad = max(0, min_x - pad_x)
    max_x_pad = max_x + pad_x + 1
    
    return (slice(min_z_pad, max_z_pad), 
            slice(min_y_pad, max_y_pad), 
            slice(min_x_pad, max_x_pad))

def global_surface_26n(mask_3d: np.ndarray) -> np.ndarray:
    print("Computing global surface mask...")
    
    structure = generate_binary_structure(3, 3)  # 26-connectivity
    binary_mask = (mask_3d > 0).astype(bool)
    eroded = binary_erosion(binary_mask, structure=structure)
    global_surface = binary_mask & ~eroded
    
    print(f"Global surface mask computed: {global_surface.sum()} surface voxels")
    return global_surface

def all_cell_bboxes(mask_3d: np.ndarray) -> Dict[int, Tuple[slice, slice, slice]]:
    print("Computing bounding boxes for all cells...")
    
    unique_ids = set(np.unique(mask_3d))
    unique_ids.discard(0)
    
    bboxes = get_bounding_boxes_3d(mask_3d, unique_ids)
    
    print(f"Bounding boxes computed for {len(bboxes)} cells")
    return bboxes

def precompute_global_surface_and_halo_bboxes(
    mask_3d: np.ndarray, 
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[np.ndarray, Dict[int, Tuple[slice, slice, slice]]]:
    print("Pre-computing global surface and halo-extended bounding boxes...")
    
    # Step 1: Compute global surface mask once
    global_surface = global_surface_26n(mask_3d)
    
    # Step 2: Get all bounding boxes
    all_bboxes = all_cell_bboxes(mask_3d)
    
    # Step 3: Precompute halo-extended bounding boxes
    print("Pre-computing halo-extended bounding boxes...")
    all_bboxes_with_halo = {}
    
    # Calculate halo padding
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    for cell_id, bbox in all_bboxes.items():
        slice_z, slice_y, slice_x = bbox
        
        # Create extended bounding box with halo
        z_start = max(0, slice_z.start - pad_z)
        z_stop = min(mask_3d.shape[0], slice_z.stop + pad_z)
        y_start = max(0, slice_y.start - pad_y)
        y_stop = min(mask_3d.shape[1], slice_y.stop + pad_y)
        x_start = max(0, slice_x.start - pad_x)
        x_stop = min(mask_3d.shape[2], slice_x.stop + pad_x)
        
        all_bboxes_with_halo[cell_id] = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Pre-computed halo-extended bounding boxes for {len(all_bboxes_with_halo)} cells")
    print(f"Using only halo-extended bboxes for all operations")
    
    return global_surface, all_bboxes_with_halo, all_bboxes

## Surface-based neighbor identification

def cell_neighbor_candidate_centroid_distance_kdtree(conn: sqlite3.Connection, 
                                                   cell_id: int,
                                                   radius_um: float = 75.0,
                                                   voxel_size_um: tuple = (0.56, 0.28, 0.28)) -> Set[int]:
    from scipy.spatial import cKDTree
    
    if isinstance(cell_id, np.ndarray):
        cell_id = int(cell_id.ravel()[0])
    else:
        cell_id = int(cell_id)

    metadata_df = get_cells_dataframe(conn)

    if cell_id not in metadata_df['cell_id'].values:
        raise ValueError(f"Cell ID {cell_id} not found in database")

    coords = metadata_df[['centroid_z', 'centroid_y', 'centroid_x']].values
    scaled_coords = coords * np.array(voxel_size_um)
    kdtree = cKDTree(scaled_coords)

    target_idx = metadata_df.index[metadata_df['cell_id'] == cell_id][0]
    target_point = scaled_coords[target_idx]
    indices = kdtree.query_ball_point(target_point, r=radius_um)
    candidate_neighbor_ids = set(metadata_df.iloc[i]['cell_id'] for i in indices if metadata_df.iloc[i]['cell_id'] != cell_id)

    return candidate_neighbor_ids

def find_touching_neighbors_direct_adjacency(mask_3d: np.ndarray, all_bboxes: dict, n_jobs: int = 1) -> set:
    import numpy as np
    from tqdm import tqdm

    print("Finding touching neighbors...")

    labels = mask_3d
    z_dim, y_dim, x_dim = labels.shape

    touching_pairs: set = set()

    # Helper function: add pairs from two same-shaped arrays
    def add_pairs(a: np.ndarray, b: np.ndarray) -> None:
        diff_mask = (a != b)
        if not diff_mask.any():
            return
        a_nz = a[diff_mask]
        b_nz = b[diff_mask]

        nz_mask = (a_nz != 0) & (b_nz != 0)
        if not nz_mask.any():
            return
        a_nz = a_nz[nz_mask]
        b_nz = b_nz[nz_mask]

        minv = np.minimum(a_nz, b_nz)
        maxv = np.maximum(a_nz, b_nz)
        touching_pairs.update(zip(minv.astype(np.int64).tolist(), maxv.astype(np.int64).tolist()))

    # Z-axis face adjacency: compare slice z with z+1
    for z in tqdm(range(z_dim - 1), desc="6-conn touching: Z faces", ncols=100):
        a = labels[z + 1, :, :]
        b = labels[z, :, :]
        add_pairs(a, b)

    # Y-axis face adjacency: within each z-slice, compare row y with y+1
    for z in tqdm(range(z_dim), desc="6-conn touching: Y faces", ncols=100):
        s = labels[z]
        a = s[1:, :]
        b = s[:-1, :]
        add_pairs(a, b)

    # X-axis face adjacency: within each z-slice, compare col x with x+1
    for z in tqdm(range(z_dim), desc="6-conn touching: X faces", ncols=100):
        s = labels[z]
        a = s[:, 1:]
        b = s[:, :-1]
        add_pairs(a, b)

    print(f"Identified {len(touching_pairs)} touching neighbor pairs.")
    return touching_pairs

def find_all_neighbors_by_surface_distance_optimized(
    mask_3d: np.ndarray,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    n_jobs: int = 1
) -> pd.DataFrame:
    required_cols = ['CellID', 'phenotype', 'Z_centroid', 'Y_centroid', 'X_centroid']
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    cell_type_map = dict(zip(metadata_df['CellID'], metadata_df['phenotype']))
    
    print(f"Step 1: Finding touching cells...")
    touching_pairs = find_touching_neighbors_direct_adjacency(mask_3d, n_jobs)
    
    neighbor_data = []
    for cell_a_id, cell_b_id in touching_pairs:
        cell_a_type = cell_type_map.get(cell_a_id, 'Unknown')
        cell_b_type = cell_type_map.get(cell_b_id, 'Unknown')
        
        neighbor_data.append({
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'cell_a_type': cell_a_type,
            'cell_b_type': cell_b_type,
            'surface_distance_um': 0.0  # Touching cells have distance = 0
        })
    
    touching_count = len(neighbor_data)
    print(f"Found {touching_count} touching neighbor pairs")
    
    # If max_distance_um = 0.0, return only touching cells
    if max_distance_um == 0.0:
        print(f"Returning only touching cells (max_distance_um = 0.0)")
        neighbor_df = pd.DataFrame(neighbor_data)
        return neighbor_df
    
    print(f"Step 2: Finding non-touching neighbors within {max_distance_um} μm...")
    print(f"Using centroid pre-filter radius: {centroid_prefilter_radius_um} μm")
    
    from joblib import Parallel, delayed
    
    centroids = metadata_df[['Z_centroid', 'Y_centroid', 'X_centroid']].values
    scaled_centroids = centroids * np.array(voxel_size_um)
    kdtree = cKDTree(scaled_centroids)
    cell_ids = metadata_df['CellID'].values
    
    touching_pairs_set = touching_pairs
    
    def process_cell_pair_optimized(cell_a_idx):
        cell_a_id = cell_ids[cell_a_idx]
        cell_a_centroid = scaled_centroids[cell_a_idx]
        
        candidate_indices = kdtree.query_ball_point(
            cell_a_centroid, 
            r=centroid_prefilter_radius_um
        )
        
        neighbors = []
        for cell_b_idx in candidate_indices:
            if cell_b_idx <= cell_a_idx:  # Avoid duplicate pairs
                continue
                
            cell_b_id = cell_ids[cell_b_idx]
            
            if (cell_a_id, cell_b_id) in touching_pairs_set or (cell_b_id, cell_a_id) in touching_pairs_set:
                continue # Skip pair if already touching
            
            try:
                distance = compute_surface_to_surface_distance_optimized(
                    mask_3d, cell_a_id, cell_b_id, voxel_size_um, max_distance_um
                )
                
                if distance <= max_distance_um:
                    neighbors.append({
                        'cell_a_id': cell_a_id,
                        'cell_b_id': cell_b_id,
                        'cell_a_type': metadata_df.iloc[cell_a_idx]['phenotype'],
                        'cell_b_type': metadata_df.iloc[cell_b_idx]['phenotype'],
                        'surface_distance_um': distance
                    })
            except Exception as e:
                print(f"Error computing optimized distance for pair ({cell_a_id}, {cell_b_id}): {e}")
                continue
        
        return neighbors
    
    # Optimization: parallel processing
    print(f"Processing {len(cell_ids)} cells for non-touching neighbors...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_pair_optimized)(i) for i in tqdm(range(len(cell_ids)), desc="Finding non-touching neighbors with optimized EDT")
    )
    
    for cell_neighbors in results:
        neighbor_data.extend(cell_neighbors)
    
    neighbor_df = pd.DataFrame(neighbor_data)
    
    total_count = len(neighbor_df)
    non_touching_count = total_count - touching_count
    
    print(f"Neighbor detection complete:")
    print(f"  - Touching neighbors: {touching_count}")
    print(f"  - Non-touching neighbors within {max_distance_um} μm: {non_touching_count}")
    print(f"  - Total neighbors: {total_count}")
    
    if non_touching_count > 0:
        non_touching_df = neighbor_df[neighbor_df['surface_distance_um'] > 0.0]
        print(f"Distance statistics for non-touching neighbors:")
        print(f"  Min distance: {non_touching_df['surface_distance_um'].min():.3f} μm")
        print(f"  Max distance: {non_touching_df['surface_distance_um'].max():.3f} μm")
        print(f"  Mean distance: {non_touching_df['surface_distance_um'].mean():.3f} μm")
    
    return neighbor_df
