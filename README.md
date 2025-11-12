# InterSCellar
[![PyPI](https://img.shields.io/pypi/v/interscellar?logo=pypi&logoColor=blue)](https://pypi.org/project/interscellar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**InterSCellar** is a Python package for surface-Based cell neighborhood and interaction volume analysis in 3D spatial omics.

![Package workflow](docs/package_workflow.jpeg)

## Installation

Install package:
```sh
pip install interscellar
```

## Usage

Import:
```sh
import interscellar
```

3D Pipeline:
(1) Cell Neighbor Detection & Graph Construction
```sh
neighbors_3d, adata, conn = interscellar.find_cell_neighbors_3d(
    ome_zarr_path="data/segmentation.zarr",
    metadata_csv_path="data/cell_metadata.csv",
    max_distance_um=0.5,
    voxel_size_um=(0.56, 0.28, 0.28),
    output_csv="results/neighbors_3d.csv",
    output_anndata="results/neighbors_3d.h5ad",
    db_path="results/cell_neighbor_graph.db",
    n_jobs=4
)
```

(2) Interscellar Volume Computation
```sh
volumes_3d, adata, conn = interscellar.compute_interscellar_volumes_3d(
    ome_zarr_path="data/segmentation.zarr",
    neighbor_pairs_csv="results/neighbors_3d.csv",
    neighbor_db_path="results/cell_neighbor_graph.db",   # helps auto-locate pickles
    voxel_size_um=(0.56, 0.28, 0.28),
    output_csv="results/volumes.csv",
    output_anndata="results/volumes.h5ad",
    output_mesh_zarr="results/volumes_mesh.zarr",
    db_path="results/interscellar_volumes.db",
    max_distance_um=3.0,
    intracellular_threshold_um=1.0,
    n_jobs=4
)
```

2D Pipeline:
(1) Cell Neighbor Detection & Graph Construction
```sh
neighbors_2d, adata, conn = interscellar.find_all_neighbors_2d(
    polygon_json_path="data/cell_polygons.json",
    metadata_csv_path="data/cell_metadata.csv",
    max_distance_um=1.0,                 # 0.0 => touching cells only
    output_csv="results/neighbors_2d.csv",
    output_anndata="results/neighbors_2d.h5ad",
    n_jobs=4
)
```
