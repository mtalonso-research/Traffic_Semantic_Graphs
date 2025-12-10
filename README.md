# Traffic Semantic Graphs

<img src="figures/project_overview.png" alt="project overview">

## ⚙️ Setup

### 1. Clone the Main Repository

First, clone this main repository to your local machine:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Clone External Dependencies

This project depends on three external Git repositories that must be cloned into specific local directories. These directories are intentionally ignored by the main project's Git repository (via `.gitignore`) to keep the repository lightweight.

Clone each of the following repositories:

```bash
# Clone nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git nuplan-devkit

# Clone ml-depth-pro
git clone https://github.com/apple/ml-depth-pro.git ml-depth-pro

# Clone deeplabs
git clone https://github.com/sunggukcha/deeplabs.git lib/deeplabs
```

### 3. Create Conda Environments

This project utilizes two separate Conda environments: `nuplan` and `sem_graphs`. The environment definitions are provided in the `nuplan_env.yml` and `sem_graphs_env.yml` files.

Ensure you have Conda installed and initialized in your shell (`conda init`, then restart your terminal if needed).

#### `nuplan` Environment

The `nuplan` environment is specifically used for running the NuPlan data processing script (`1B_nup_processing.py`).

1.  **Create the environment:**
    ```bash
    conda env create -f nuplan_env.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate nuplan
    ```
3.  **Install Hardware-Specific Packages:**
    After activating the environment, install PyTorch and its related packages. Please visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to find the correct installation command for your specific system (OS, package manager, CUDA version). You will need to install `torch`, `pytorch-lightning`, and `torchmetrics`.

#### `sem_graphs` Environment

The `sem_graphs` environment is used for all other scripts in the pipeline.

1.  **Create the environment:**
    ```bash
    conda env create -f sem_graphs_env.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate sem_graphs
    ```
3.  **Install Hardware-Specific Packages:**
    After activating the environment, install PyTorch, PyG (PyTorch Geometric), and related packages.
    *   First, visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to find the correct installation command for `torch`, `torchvision`, and `torchaudio` for your system.
    *   Then, visit the [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for instructions on how to install `pyg` correctly for your new PyTorch version.

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

The `1A_l2d_processing.py` script processes the L2D dataset.

**Usage:**
```bash
python -m scripts.1A_l2d_processing [-h] [--min_ep MIN_EP] [--max_ep MAX_EP] [--download] [--process_tabular] [--add_tags] [--process_frames] [--process_lanes] [--generate_graphs] [--all]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--min_ep MIN_EP`: Minimum episode number to process.
- `--max_ep MAX_EP`: Maximum episode number to process.
- `--download`: Run data download step.
- `--process_tabular`: Run tabular data processing step.
- `--add_tags`: Run tag processing step.
- `--process_frames`: Run frame processing step.
- `--process_lanes`: Run lane processing step.
- `--generate_graphs`: Run graph generation step.
- `--all`: Run all steps (default if no flags are set).

To run the entire L2D processing pipeline, use the following command:
```bash
python -m scripts.1A_l2d_processing --all
```

### 1B: NuPlan Data Processing

The `1B_nup_processing.py` script processes the NuPlan dataset.

**Usage:**
```bash
python -m scripts.1B_nup_processing [-h] [--city CITY] [--file_min FILE_MIN] [--file_max FILE_MAX] [--episodes EPISODES [EPISODES ...]] [--load] [--latlon] [--weather] [--weather_codes] [--temporal] [--tags]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--city CITY`: City to process (boston or pittsburgh).
- `--file_min FILE_MIN`: Minimum DB file index to process (inclusive).
- `--file_max FILE_MAX`: Maximum DB file index to process (exclusive). Use 'none' for all after file_min.
- `--episodes EPISODES [EPISODES ...]`: Episodes to Process.
- `--load`: Run only the data loading step.
- `--latlon`: Run only the lat/lon addition step.
- `--weather`: Run only the weather enrichment step.
- `--weather_codes`: Run only the weather code replacement step.
- `--temporal`: Run only the temporal feature addition step.
- `--tags`: Run only the tag extraction step.

To run the NuPlan processing for a specific city (e.g., Boston) and a specific step (e.g., tags), use the following command:
```bash
python -m scripts.1B_nup_processing --city boston --tags
```

To run all steps, you can omit the step-specific flags:
```bash
python -m scripts.1B_nup_processing --city boston
```

### 1C: Final Post-Processing

The `1C_final_processing.py` script performs final post-processing on the generated graph data.

**Usage:**
```bash
python -m scripts.1C_final_processing [-h] [--process_l2d] [--process_nuplan_boston] [--process_nuplan_pittsburgh]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--process_l2d`: Run L2D final processing
- `--process_nuplan_boston`: Run nuPlan Boston final processing
- `--process_nuplan_pittsburgh`: Run nuPlan Pittsburgh final processing

To run the final processing for a specific dataset, use the corresponding flag. For example, to process the L2D dataset:
```bash
python -m scripts.1C_final_processing --process_l2d
```

## Phase 2: Frame Encoding

This phase involves encoding the image frames from the L2D dataset using a pretrained model to generate frame embeddings.

### 2A: Frame Encoding

The `2_frame_encoding.py` script encodes the frames from the L2D dataset.

**Usage:**
```bash
python -m scripts.2_frame_encoding [-h] [--model_path MODEL_PATH] [--output_dir OUTPUT_DIR] [--frames_root FRAMES_ROOT] [--run_encoding] [--all] [--min_ep MIN_EP] [--max_ep MAX_EP]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--model_path MODEL_PATH`: Path to the pretrained model.
- `--output_dir OUTPUT_DIR`: Directory to save the embeddings.
- `--frames_root FRAMES_ROOT`: Root directory of the frames.
- `--run_encoding`: Run the frame encoding step.
- `--all`: Run all steps (default if no flags are set).
- `--min_ep MIN_EP`: Minimum episode ID to process.
- `--max_ep MAX_EP`: Maximum episode ID to process.

To run the frame encoding pipeline, use the following command:
```bash
python -m scripts.2_frame_encoding --all
```

## Phase 3: Risk Analysis

This phase involves running the risk analysis on the processed data to generate risk scores for each episode.

### 3A: Risk Analysis

The `3_risk_analysis.py` script runs the risk analysis on the graph data.

**Usage:**
```bash
python -m scripts.3_risk_analysis [-h] [--run_analysis] [--run_on_all_episodes] [--extract_large_risk_eps] [--extract_low_risk_eps] [--risk_statistics] [--dataset DATASET] [--input_directory INPUT_DIRECTORY] [--output_directory OUTPUT_DIRECTORY] [--risk_csv_dir RISK_CSV_DIR] [--output_filename OUTPUT_FILENAME] [--num_episodes NUM_EPISODES] [--threshold THRESHOLD] [--columns COLUMNS [COLUMNS ...]]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--run_analysis`: Run the full risk analysis and generate a CSV.
- `--run_on_all_episodes`: Run risk analysis on all episodes in a directory.
- `--extract_large_risk_eps`: Extract episodes with risk above a threshold.
- `--extract_low_risk_eps`: Extract episodes with risk below a threshold.
- `--risk_statistics`: Show statistics for specified columns.
- `--dataset DATASET`: Dataset name to use default directory structures (e.g., 'L2D', 'NUP').
- `--input_directory INPUT_DIRECTORY`: Override default input directory for graph JSON files.
- `--output_directory OUTPUT_DIRECTORY`: Override default output directory for the CSV file.
- `--risk_csv_dir RISK_CSV_DIR`: Override default path to the risk_data.csv file for analysis.
- `--output_filename OUTPUT_FILENAME`: Output filename for the JSON file (e.g., 'risk_results.json').
- `--num_episodes NUM_EPISODES`: Number of episodes to process (for --run_analysis).
- `--threshold THRESHOLD`: Risk threshold for episode extraction.
- `--columns COLUMNS [COLUMNS ...]`: List of columns for statistics.

To run the risk analysis on the L2D dataset, use the following command:
```bash
python -m scripts.3_risk_analysis --run_on_all_episodes --dataset L2D --output_filename l2d_max_risks.json
```

## Phase 4: Risk Prediction

This phase involves training a model to predict risk based on the processed data.

### 4A: Risk Prediction

The `4A_risk_prediction.py` script trains a risk prediction model.

**Usage:**
```bash
python -m scripts.4A_risk_prediction [-h] [--input_directory INPUT_DIRECTORY] [--val_dir VAL_DIR] [--val_risk_scores_path VAL_RISK_SCORES_PATH] [--l2d] [--nup] [--with_side_information] [--mode MODE] [--hidden_dim HIDDEN_DIM] [--embed_dim EMBED_DIM] [--num_encoder_layers NUM_ENCODER_LAYERS] [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--val_fraction VAL_FRACTION] [--num_epochs NUM_EPOCHS] [--lr LR] [--weight_decay WEIGHT_DECAY] [--train] [--evaluate] [--save_annotations] [--prediction_mode {regression,classification}] [--sweep] [--best_model_path BEST_MODEL_PATH] [--seed SEED] [--load_config LOAD_CONFIG]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--input_directory INPUT_DIRECTORY`: Input directory for graph data.
- `--val_dir VAL_DIR`: Directory for validation graphs.
- `--val_risk_scores_path VAL_RISK_SCORES_PATH`: Path to validation risk scores.
- `--l2d`: Process L2D dataset.
- `--nup`: Process NuPlan dataset.
- `--with_side_information`: Load side information for L2D dataset.
- `--mode MODE`: Graph dataset mode.
- `--hidden_dim HIDDEN_DIM`: Dimension of hidden layers.
- `--embed_dim EMBED_DIM`: Dimension of latent embeddings.
- `--num_encoder_layers NUM_ENCODER_LAYERS`: Number of encoder layers.
- `--activation ACTIVATION`: Activation function.
- `--dropout_rate DROPOUT_RATE`: Dropout rate.
- `--batch_size BATCH_SIZE`: Batch size.
- `--num_workers NUM_WORKERS`: Number of DataLoader workers.
- `--val_fraction VAL_FRACTION`: Fraction of data for validation.
- `--num_epochs NUM_EPOCHS`: Number of training epochs.
- `--lr LR`: Learning rate.
- `--weight_decay WEIGHT_DECAY`: Weight decay.
- `--train`: Train the model.
- `--evaluate`: Evaluate the model.
- `--save_annotations`: Save evaluation losses to a file.
- `--prediction_mode {regression,classification}`: Risk prediction mode.
- `--sweep`: Run a wandb sweep.
- `--best_model_path BEST_MODEL_PATH`: Path to save/load the best model checkpoint.
- `--seed SEED`: Random seed.
- `--load_config LOAD_CONFIG`: Path to a YAML config file with default arguments.

To train the risk prediction model on the L2D dataset, use the following command:
```bash
python -m scripts.4A_risk_prediction --l2d --train --evaluate --save_annotations
```

### 4B: Graph Encoding

The `4B_graph_encoding.py` script trains a graph autoencoder to generate graph embeddings.

**Usage:**
```bash
python -m scripts.4B_graph_encoding [-h] [--dataset DATASET] [--base_dataset_dir BASE_DATASET_DIR] [--train_encoder] [--evaluate] [--mode MODE] [--hidden_dim HIDDEN_DIM] [--embed_dim EMBED_DIM] [--num_encoder_layers NUM_ENCODER_LAYERS] [--num_decoder_layers NUM_DECODER_LAYERS] [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE] [--side_info_path SIDE_INFO_PATH] [--risk_scores_path RISK_SCORES_PATH] [--l2d_risk_scores_path L2D_RISK_SCORES_PATH] [--nuplan_risk_scores_path NUPLAN_RISK_SCORES_PATH] [--node_features_to_exclude NODE_FEATURES_TO_EXCLUDE [NODE_FEATURES_TO_EXCLUDE ...]] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--pin_memory] [--val_fraction VAL_FRACTION] [--num_epochs NUM_EPOCHS] [--lr LR] [--weight_decay WEIGHT_DECAY] [--kl_weight KL_WEIGHT] [--best_model_path BEST_MODEL_PATH] [--output_dir OUTPUT_DIR] [--seed SEED] [--use_wandb] [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--wandb_run_name WANDB_RUN_NAME] [--wandb_group WANDB_GROUP] [--wandb_mode {online,offline,disabled}] [--config CONFIG]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--dataset DATASET`: Dataset name (used in paths under ./data/).
- `--base_dataset_dir BASE_DATASET_DIR`: Base directory for graph datasets.
- `--train_encoder`: Whether to train the frame encoder.
- `--evaluate`: Whether to run evaluation / embedding extraction.
- `--mode MODE`: Graph dataset mode.
- `--hidden_dim HIDDEN_DIM`: Dimension of hidden layers.
- `--embed_dim EMBED_DIM`: Dimension of latent embeddings.
- `--num_encoder_layers NUM_ENCODER_LAYERS`: Number of encoder layers.
- `--num_decoder_layers NUM_DECODER_LAYERS`: Number of decoder layers.
- `--activation ACTIVATION`: Activation function.
- `--dropout_rate DROPOUT_RATE`: Dropout rate.
- `--side_info_path SIDE_INFO_PATH`: Path to side information file.
- `--risk_scores_path RISK_SCORES_PATH`: Path to risk scores JSON file.
- `--l2d_risk_scores_path L2D_RISK_SCORES_PATH`: Path to L2D risk scores JSON file.
- `--nuplan_risk_scores_path NUPLAN_RISK_SCORES_PATH`: Path to NuPlan risk scores JSON file.
- `--node_features_to_exclude NODE_FEATURES_TO_EXCLUDE [NODE_FEATURES_TO_EXCLUDE ...]`: List of node features to exclude.
- `--batch_size BATCH_SIZE`: Batch size (episodes per batch).
- `--num_workers NUM_WORKERS`: Number of DataLoader workers.
- `--pin_memory`: Use pin_memory in DataLoader (useful on CUDA).
- `--val_fraction VAL_FRACTION`: Fraction of data to use for validation.
- `--num_epochs NUM_EPOCHS`: Number of training epochs.
- `--lr LR`: Learning rate for Adam optimizer.
- `--weight_decay WEIGHT_DECAY`: Weight decay for Adam optimizer.
- `--kl_weight KL_WEIGHT`: Weight for the KL divergence loss.
- `--best_model_path BEST_MODEL_PATH`: Path to save / load the best model checkpoint.
- `--output_dir OUTPUT_DIR`: Directory to save extracted embeddings.
- `--seed SEED`: Random seed for numpy and torch.
- `--use_wandb`: Enable Weights & Biases logging and sweeps.
- `--wandb_project WANDB_PROJECT`: wandb project name.
- `--wandb_entity WANDB_ENTITY`: wandb entity (user or team).
- `--wandb_run_name WANDB_RUN_NAME`: Optional wandb run name.
- `--wandb_group WANDB_GROUP`: Optional group name for this run.
- `--wandb_mode {online,offline,disabled}`: wandb mode.
- `--config CONFIG`: Optional YAML config file to override/default arguments.

To train the graph encoder, use the following command:
```bash
python -m scripts.4B_graph_encoding --train_encoder --evaluate
```

### 4C: Risk Prediction from Embeddings

The `4C_risk_prediction_from_embeddings.py` script trains a risk prediction head on pre-trained graph embeddings.

**Usage:**
```bash
python -m scripts.4C_risk_prediction_from_embeddings [-h] [--embedding_dir EMBEDDING_DIR] [--nuplan_embedding_file NUPLAN_EMBEDDING_FILE] [--l2d_embedding_file L2D_EMBEDDING_FILE] [--nuplan_risk_scores_path NUPLAN_RISK_SCORES_PATH] [--l2d_risk_scores_path L2D_RISK_SCORES_PATH] [--hidden_dim HIDDEN_DIM] [--prediction_mode {regression,classification}] [--batch_size BATCH_SIZE] [--val_fraction VAL_FRACTION] [--num_epochs NUM_EPOCHS] [--lr LR] [--weight_decay WEIGHT_DECAY] [--train] [--evaluate] [--save_annotations] [--side_info_4b] [--best_model_path BEST_MODEL_PATH]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--embedding_dir EMBEDDING_DIR`: Directory where graph embeddings are stored.
- `--nuplan_embedding_file NUPLAN_EMBEDDING_FILE`: NuPlan embedding file name.
- `--l2d_embedding_file L2D_EMBEDDING_FILE`: L2D embedding file name.
- `--nuplan_risk_scores_path NUPLAN_RISK_SCORES_PATH`: Path to NuPlan risk scores JSON file.
- `--l2d_risk_scores_path L2D_RISK_SCORES_PATH`: Path to L2D validation risk scores JSON file.
- `--hidden_dim HIDDEN_DIM`: Dimension of hidden layers.
- `--prediction_mode {regression,classification}`: Risk prediction mode.
- `--batch_size BATCH_SIZE`: Batch size.
- `--val_fraction VAL_FRACTION`: Fraction of data for validation.
- `--num_epochs NUM_EPOCHS`: Number of training epochs.
- `--lr LR`: Learning rate.
- `--weight_decay WEIGHT_DECAY`: Weight decay.
- `--train`: Train the model.
- `--evaluate`: Evaluate the model.
- `--save_annotations`: Save evaluation losses to a file.
- `--side_info_4b`: Flag to indicate that the embeddings from 4B were trained with side information.
- `--best_model_path BEST_MODEL_PATH`: Path to save/load the best model checkpoint.

To train the risk prediction head from embeddings, use the following command:
```bash
python -m scripts.4C_risk_prediction_from_embeddings --train --evaluate --save_annotations
```

## Phase 5: Visualization

This phase involves visualizing the data and results.

### 5A: Visualization

The `visualizer.py` script provides several visualization options.

**Usage:**
```bash
python -m scripts.visualizer [-h] [--dataset_path DATASET_PATH] [--episode EPISODE] [--frame FRAME] [--graph_visualizer] [--scene_visualizer] [--histogram HISTOGRAM]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--dataset_path DATASET_PATH`: Path to the dataset directory.
- `--episode EPISODE`: Episode number to visualize.
- `--frame FRAME`: Frame number for scene visualizer.
- `--graph_visualizer`: Run the combined graph visualizer.
- `--scene_visualizer`: Run the original scene visualizer.
- `--histogram HISTOGRAM`: Plot a histogram of a feature (e.g., 'ego-vx' or 'ego-all').

**Examples:**

To run the graph visualizer for a specific episode:
```bash
python -m scripts.visualizer --graph_visualizer --episode 1
```

To run the scene visualizer for a specific episode and frame:
```bash
python -m scripts.visualizer --scene_visualizer --episode 1 --frame 10
```

To plot a histogram of a specific feature:
```bash
python -m scripts.visualizer --histogram ego-vx
```
