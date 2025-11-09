# Import

import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
import cv2
import csv
import sqlite3
import os
import pickle
import math
import json

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Cell surface precomputation

def build_global_mask_2d(polygon_mask: dict) -> Tuple[np.ndarray, Tuple[int, int], dict]:
    print("Building global 2D mask...")
    
    all_points = np.concatenate([np.array(pts) for pts in polygon_mask.values()])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    mask_shape = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)
    global_mask = np.zeros(mask_shape, dtype=np.int32)
    cell_id_mapping = {cid: i+1 for i, cid in enumerate(polygon_mask.keys())}
    
    for cid, pts in polygon_mask.items():
        mapped_id = cell_id_mapping[cid]
        pts = np.array(pts, dtype=np.float32)
        
        shifted_pts = pts - np.array([min_x, min_y])
        
        int_pts = shifted_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(global_mask, [int_pts], mapped_id)
    
    print(f"Global 2D mask generated: shape {global_mask.shape}, {len(polygon_mask)} cells")
    return global_mask, mask_shape, cell_id_mapping

def build_global_mask_2d_with_mapping(polygon_mask: dict, cell_id_mapping: dict) -> Tuple[np.ndarray, Tuple[int, int]]:
    print("Building global 2D mask...")
    
    all_points = np.concatenate([np.array(pts) for pts in polygon_mask.values()])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    mask_shape = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)
    global_mask = np.zeros(mask_shape, dtype=np.int32)
    
    for cid, pts in polygon_mask.items():
        if cid not in cell_id_mapping:
            continue
        mapped_id = cell_id_mapping[cid]
        pts = np.array(pts, dtype=np.float32)
        
        shifted_pts = pts - np.array([min_x, min_y])
        
        int_pts = shifted_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(global_mask, [int_pts], mapped_id)
    
    print(f"Global mask generated: shape {global_mask.shape}, {len(polygon_mask)} cells")
    return global_mask, mask_shape

def get_bounding_boxes_2d(global_mask: np.ndarray, unique_ids: set) -> dict:
    y, x = np.nonzero(global_mask)
    cell_ids = global_mask[y, x]
    df = pd.DataFrame({'cell_id': cell_ids, 'y': y, 'x': x})
    bbox = {}
    grouped = df.groupby('cell_id')
    for cell_id, group in grouped:
        if cell_id == 0:
            continue
        miny, maxy = group['y'].min(), group['y'].max() + 1
        minx, maxx = group['x'].min(), group['x'].max() + 1
        bbox[cell_id] = (slice(miny, maxy), slice(minx, maxx))
    return bbox

def compute_bounding_box_with_halo_2d(
    surface_a: np.ndarray,
    max_distance_um: float,
    pixel_size_um: float
) -> Tuple[slice, slice]:
    y_coords, x_coords = np.where(surface_a)
    
    if len(y_coords) == 0:
        return None
    
    min_y, max_y = y_coords.min(), y_coords.max() + 1
    min_x, max_x = x_coords.min(), x_coords.max() + 1
    
    pad_y = math.ceil(max_distance_um / pixel_size_um)
    pad_x = math.ceil(max_distance_um / pixel_size_um)
    
    min_y_pad = max(0, min_y - pad_y)
    max_y_pad = max_y + pad_y + 1
    min_x_pad = max(0, min_x - pad_x)
    max_x_pad = max_x + pad_x + 1
    
    return (slice(min_y_pad, max_y_pad), slice(min_x_pad, max_x_pad))

def global_surface_2d(global_mask: np.ndarray) -> np.ndarray:
    print("Computing global surface mask...")
    
    structure = generate_binary_structure(2, 2)
    binary_mask = (global_mask > 0).astype(bool)
    eroded = binary_erosion(binary_mask, structure=structure)    
    global_surface = binary_mask & ~eroded
    
    print(f"Global surface mask computed: {global_surface.sum()} surface pixels")
    return global_surface

def all_cell_bboxes_2d(global_mask: np.ndarray) -> Dict[int, Tuple[slice, slice]]:
    print("Computing bounding boxes for all cells in single sweep...")
    
    unique_ids = set(np.unique(global_mask))
    unique_ids.discard(0)
    
    bboxes = get_bounding_boxes_2d(global_mask, unique_ids)
    
    print(f"Bounding boxes computed for {len(bboxes)} cells")
    return bboxes

def precompute_global_surface_and_halo_bboxes_2d(
    global_mask: np.ndarray, 
    max_distance_um: float,
    pixel_size_um: float
) -> Tuple[np.ndarray, Dict[int, Tuple[slice, slice]]]:
    print("Pre-computing global surface and halo-extended bounding boxes...")
    
    global_surface = global_surface_2d(global_mask)

    all_bboxes = all_cell_bboxes_2d(global_mask)
    
    print("Pre-computing halo-extended bounding boxes...")
    all_bboxes_with_halo = {}
    
    pad_y = math.ceil(max_distance_um / pixel_size_um)
    pad_x = math.ceil(max_distance_um / pixel_size_um)
    
    for cell_id, bbox in all_bboxes.items():
        slice_y, slice_x = bbox
        
        y_start = max(0, slice_y.start - pad_y)
        y_stop = min(global_mask.shape[0], slice_y.stop + pad_y)
        x_start = max(0, slice_x.start - pad_x)
        x_stop = min(global_mask.shape[1], slice_x.stop + pad_x)
        
        all_bboxes_with_halo[cell_id] = (slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Pre-computed halo-extended bounding boxes for {len(all_bboxes_with_halo)} cells")
    
    return global_surface, all_bboxes_with_halo, all_bboxes

## Surface-based neighbor identification

def find_touching_neighbors_2d(global_mask: np.ndarray, all_bboxes: dict, n_jobs: int = 1) -> set:
    print("Finding touching neighbors...")
    
    labels = global_mask
    y_dim, x_dim = labels.shape
    
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
    
    # Y-axis face adjacency: compare row y with y+1
    for y in tqdm(range(y_dim - 1), desc="4-conn touching: Y faces", ncols=100):
        a = labels[y + 1, :]
        b = labels[y, :]
        add_pairs(a, b)
    
    # X-axis face adjacency: compare col x with x+1
    for y in tqdm(range(y_dim), desc="4-conn touching: X faces", ncols=100):
        a = labels[:, 1:]
        b = labels[:, :-1]
        add_pairs(a, b)
    
    print(f"Found {len(touching_pairs)} touching neighbor pairs")
    return touching_pairs

