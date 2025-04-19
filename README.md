# MapMyMovies - Collaborative Filtering Recommendation System

## DESCRIPTION
This package implements two recommendation system approaches:

1. **Regular Collaborative Filtering**: A traditional collaborative filtering implementation that combines both user-based and item-based approaches to provide high-quality, personalized recommendations. 
The system identifies users with similar preferences or items with similar patterns and uses these similarities to predict ratings. You can also use a hybrid approach (described in the paper) that uses a 
weighted combination of both model's predictions.

2. **Neural Collaborative Filtering (Neural Matrix Factorization)**: A deep learning approach that combines Generalized Matrix Factorization and Multi-Layer Perceptron to model user-item interactions, offering improved recommendation performance with non-linear transformations. 
Our implementation includes diversity enhancement techniques to address popularity bias.

**Dataset**: For this repo, the MovieLens 100k small dataset is provided for user testing purposes. Both implementations are evaluated on the MovieLens dataset and include metrics for accuracy (RMSE, MAE). 
The neural model additional contains diversity metrics (coverage, entropy, and overlap). 

### Interactive Visualization
The system includes an interactive network visualization built with Plotly and Cytoscape. Each model (User-Based, Item-Based, and Neural) has its own unique network visualization, allowing users to understand the 
specific workings behind their recommendations. The visualization creates a transparent view of how recommendations are generated, showing connections between users, items, and predicted ratings.

## INSTALLATION

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (for Neural CF)

### Setting up the environment
```bash
# Create and activate a conda environment
conda create -n recommender python=3.8
conda activate recommender

# Install dependencies for Regular CF
cd CODE/regular-cf
pip install -r requirements.txt

# Install dependencies for Neural CF
cd CODE/neural-cf
pip install -r requirements.txt
```

## EXECUTION
### Running Regular Collaborative Filtering
This script generates movie recommendations using user-based, item-based, or hybrid collaborative filtering.
Pre-computed neighbors are available and located in pickle files in the `CODE/regular-cf/data` folder.

**Example:**
```bash
cd CODE/regular-cf
# Example: general training
python run.py [OPTIONS]
# Example: generate 10 hybrid recommendations for user 10 and save neighborhoods
python main.py --data_dir data --user_id 10 --method hybrid --num_recs 10 --save_neighbors
```

Below are optional arguments:
| Flag             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--data_dir`     | Path to directory containing `ratings.csv` and `movies.csv` (default: `data/`) |
| `--user_id`      | User ID to generate recommendations for (if omitted, a random user is selected) |
| `--method`       | Recommendation method: `ubcf`, `ibcf`, or `hybrid` (default: `hybrid`)      |
| `--num_recs`     | Number of movie recommendations to generate (default: `10`)                 |
| `--top_n`        | Number of neighbors to consider when generating recommendations (default: `100`) |
| `--load_neighbors` | Load pre-computed user/item neighborhoods if available                   |
| `--save_neighbors` | Save computed neighborhoods for reuse                                    |
| `--skip_fit`     | Skip model fitting (use default alpha for hybrid method)                    |

### Running Neural Collaborative Filtering
This script generates movie recommendations using neural collaborative filtering with various configurations.
These configurations can be modified in the config dictionary located in the `train.py` file. 

Note: this model should be run using a CUDA-compatible GPU. The model was trained on 1 V100 GPU. 
Compute configurations can be found in the `run.sh` script that was submitted to the university cluster.
If a SLURM job is submitted, the training log will appear in a file called `neuralcf-train.out`. 

Below is a description of the directory: 
* `checkpoints`: latest model checkpoints, split up by explicit vs implicit feedback. For explicit, model checkpoints from the final epoch of 2 runs are provided. 
* `data`: contains MovieLens 100k files for testing purposes
* `explanations`: contains the JSON files used to visualize the model in the graph application
* `recommendations-out`: contains loss, accuracy and diversity metrics and plots, as well as configurations for 2 runs
* `runs`: contains TensorBoard log training metrics


**Example (GPU):**
```bash
sbatch run.sh
```

**Example (local):***
```bash
python train.py 
```


### Running the Visualization Application

The visualization can be run using the following code:

```bash
python graph_app.py
```

Note: This visualization was developed and tested primarily in Firefox, which we recommend for the best 
experience. Minor inconsistencies may occur when using Chrome or Safari. Instructions on how to navigate the interactive graph are 
located in the application itself and in the paper. 
