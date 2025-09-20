# Traffic Semantic Graphs

### 🚧 Current State

- **Phase 1: Graph Generation**
  - ✅ Data processing and graph generation pipeline is fully functional.
  - ✅ Speed detection output is integrated as a node feature for all vehicles. 
  - ✅ Balanced subset of the dataset (with respect to vehicle action) is ready.
  - ✅ The graphs for the two datasets have been standardized to follow the same structure.
  - 🔄 Currently integrating lane detection output as a node feature across all graphs.

- **Phase 2: Baseline Models**
  - ✅ Data loader for pytorch geometric HeteroData is complete.
  - ✅ Graphs for both datasets are effectively stored as PyG HeteroData objects
  - ✅ Initial experiments conducted using PyTorch Geometric with heterogeneous graphs and a basic MLP for downstream tasks.
  - 🔍 Currently in progress: brainstorming what to do as a downstream task.
     - Needs to work for both datasets.
     - Needs to use information from several nodes (next action or next coordinate tasks only require ego vehicle information).
  - ⏳ Still pending:
    - Incorporation of camera data.
    - Improvements to the model architecture and training pipeline.

- **Phase 3: Knowledge Distillation**
  - 🔍 Currently in progress: graph matching mechanism to align similar traffic scenes across two datasets.
     - Focusing on clustering and graph classification at the moment.


## 📁 Directory Overview

```plaintext
TRAFFIC_SEMANTIC_GRAPHS/
├── data/ # Local-only: raw and processed data (not included in repo)
│ ├── graphical/L2D/ # Graph files for L2D dataset
│ ├── processed/L2D/ # Processed tabular data
│ ├── processed_frames/L2D/ # Processed image frames (e.g., YOLO, depth outputs)
│ ├── raw/L2D/ # Raw input (images, tabular)
│ │ ├── frames/
│ │ └── tabular/
│ └── semantic_tags/L2D/ # semi-manually-generated semantic tags
├── functions/ # Python modules for data loading, processing, and visualization
│ ├── graphs_L2D.py
│ ├── load_data_L2D.py
│ ├── models_L2D.py
│ ├── process_frames_L2D.py
│ ├── process_tabular_data_L2D.py
│ ├── process_tags_L2D.py
│ ├── utils_L2D.py
│ └── visualizations_L2D.py
├── notebooks/ # Main notebooks
│ ├── graph-generation.ipynb # Pipeline for graph construction
│ └── task-experiments.ipynb # Experiments on downstream tasks using the graphs
├── isac_diagram.pdf/.png # System diagram
├── requirements.txt # Python dependencies
└── README.md # This file
```
**Note:** The actual data files are not included in the repository due to storage limitations, but steps for downloading and processing the data are included in ```graph-generation.ipynb```.
