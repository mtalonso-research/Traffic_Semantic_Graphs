# GEMINI.md

## Project Overview

This project, "Traffic Semantic Graphs," is a Python-based tool for analyzing traffic scenes and assessing risk. It processes data from various sources, including the L2D and NuPlan datasets, to generate semantic graphs that represent the relationships between different entities in a traffic scene. The core of the project is a `RiskAnalysis` class that calculates a risk score based on factors like weather, road conditions, and the behavior of other vehicles.

The project is structured into several directories:

*   `data/`: Contains raw and processed data (not included in the repository).
*   `src/`: Contains the main source code for data processing, risk analysis, and visualization.
*   `scripts/`: Contains scripts for running the data processing pipeline.
*   `nuplan-devkit/`: Contains the nuPlan devkit for working with the nuPlan dataset.

## Building and Running

### Dependencies

The project's Python dependencies are not explicitly listed in a `requirements.txt` file. However, based on the imported modules, the following libraries are required:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `Pillow`
*   `tqdm`
*   `argparse`

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib Pillow tqdm argparse
```

### Running the Data Processing Pipeline

The data processing pipeline can be run using the scripts in the `scripts/` directory. For example, to process the L2D dataset, you can run the `1A_l2d_processing.py` script:

```bash
python scripts/1A_l2d_processing.py --all
```

This will run all the processing steps, including downloading the data, processing tabular data, adding tags, processing frames, processing lanes, and generating graphs. You can also run specific steps by using the corresponding flags (e.g., `--download`, `--process_tabular`, etc.).

### Running the Risk Analysis

The risk analysis is performed by the `RiskAnalysis` class in `src/risk_analysis.py`. You can use this class in your own scripts or notebooks to analyze traffic scenes and calculate risk scores. The `risk-testing-1.ipynb` notebook provides an example of how to use the `RiskAnalysis` class.

## Development Conventions

*   **Code Style:** The code follows the PEP 8 style guide for Python code.
*   **Modularity:** The code is organized into modules and classes, with a clear separation of concerns.
*   **Documentation:** The code includes docstrings and comments to explain the purpose of functions and classes.
*   **Configuration:** The `RiskAnalysis` class uses a configuration dictionary to allow for easy customization of the risk model parameters.
