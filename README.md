# Traffic Semantic Graphs

<img src="figures/project_overview.png" alt="project overview">

## 📁 Directory Overview

```plaintext
TRAFFIC_SEMANTIC_GRAPHS/
├── data/                            # Local-only: raw and processed data (not included in repo)
│   ├── distributions/               # Data distributions
│   ├── graphical/                   # Graph files for L2D and NuPlan datasets
│   ├── graphical_final/             # Final graphical data after processing and filtering
│   ├── processed/                   # Processed tabular data
│   ├── processed_frames/            # Processed image frames (e.g., YOLO, depth outputs)
│   ├── raw/                         # Raw inputs (images, tabular)
│   ├── semantic_tags/               # Semi-manually-generated semantic tags
│   └── temporary_data/              # Temporary data storage
├── figures/                         # Project figures and visualizations
│   ├── project_overview.pdf
│   └── project_overview.png
├── ml-depth-pro/                    # Machine learning depth prediction project
│   └── checkpoints/                 # Checkpoints for ML models
├── nuplan-devkit/                   # NuPlan development kit
├── scripts/                         # Scripts for data processing and visualization
│   ├── 1A_l2d_processing.py         # L2D data processing script
│   ├── 1B_nup_processing.py         # NuPlan data processing script
│   ├── 1C_final_processing.py       # Final processing script
│   └── scene_visualizer.py          # Scene visualization script
├── src/                             # Main source code
│   ├── data_processing/             # Modules for data loading and processing
│   │   ├── filtering.py
│   │   ├── final_post_processing.py
│   │   ├── l2d_generate_graphs.py
│   │   ├── l2d_lane_processing.py
│   │   ├── l2d_load_data.py
│   │   ├── l2d_process_frames.py
│   │   ├── l2d_process_pqts.py
│   │   ├── l2d_process_tags.py
│   │   ├── nup_load_data.py
│   │   └── nup_process_jsons.py
│   ├── risk_analysis.py             # Risk analysis module
│   ├── utils.py                     # Utility functions
│   └── visualizations.py            # Visualization functions
└── README.md                        # This file

```
**Note:** The actual data files are not included in the repository due to storage limitations.

## Phase 1: Data Processing

This phase involves processing the raw data from the L2D and NuPlan datasets into a format suitable for risk analysis.

### 1A: L2D Data Processing

The `1A_l2d_processing.py` script processes the L2D dataset. It performs the following steps:
- Downloads the data
- Processes tabular data
- Adds tags
- Processes frames
- Processes lanes
- Generates graphs

To run the entire L2D processing pipeline, use the following command:

```bash
python scripts/1A_l2d_processing.py --all
```

You can also run specific steps by using the corresponding flags (e.g., `--download`, `--process_tabular`, etc.).

### 1B: NuPlan Data Processing

The `1B_nup_processing.py` script processes the NuPlan dataset. It can process data for Boston and Pittsburgh. The script can perform the following steps:
- Load data
- Add latitude and longitude
- Enrich weather features
- Replace weather codes
- Add temporal features
- Extract semantic tags

To run the NuPlan processing for a specific city (e.g., Boston) and a specific step (e.g., tags), use the following command:

```bash
python scripts/1B_nup_processing.py --city boston --tags
```

To run all steps, you can omit the step-specific flags:

```bash
python scripts/1B_nup_processing.py --city boston
```

### 1C: Final Post-Processing

The `1C_final_processing.py` script performs final post-processing on the generated graph data. This includes filtering and enriching the data.

To run the final processing for a specific dataset, use the corresponding flag. For example, to process the L2D dataset:

```bash
python scripts/1C_final_processing.py --process_l2d
```

To process the NuPlan Boston dataset:

```bash
python scripts/1C_final_processing.py --process_nuplan_boston
```

To process the NuPlan Pittsburgh dataset:

```bash
python scripts/1C_final_processing.py --process_nuplan_pittsburgh
```
