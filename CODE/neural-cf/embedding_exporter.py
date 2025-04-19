import torch
import json
from neumf import NeuMF
from data import SampleGenerator
import pandas as pd
import numpy as np
import time
import os
from json import JSONEncoder

def forward_explain_detailed(model, user_id, item_id):
    """
    Runs a full forward pass for a given user_id and item_id,
    returning a dictionary containing:
      - user_mlp_embedding: the user embedding from the MLP pathway
      - item_mlp_embedding: the item embedding from the MLP pathway
      - user_mf_embedding:  the user embedding from the MF pathway
      - item_mf_embedding:  the item embedding from the MF pathway
      - mf_product:         element-wise multiplication of the MF embeddings
      - mlp_output:         the final hidden layer output from the MLP pathway
      - final_vector:       the concatenation of mlp_output and mf_product
      - final_contributions: element-wise contributions computed as final_vector * affine_weight
      - raw_rating:         the raw sigmoid output (0-1)
      - predicted_score:    the final predicted rating (raw output multiplied by max_rating for explicit feedback)
      - affine_bias:        the bias term of the final layer
      - affine_weights:     the final layer's weights (used for computing contributions)
    """
    # Convert NumPy int64 to standard Python int to avoid JSON serialization issues
    if isinstance(user_id, np.integer):
        user_id = int(user_id)
    if isinstance(item_id, np.integer):
        item_id = int(item_id)
        
    device = torch.device("cuda" if model.config.get("use_cuda", False) else "cpu")
    # Prepare input tensors (batch size 1)
    user_idx_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
    item_idx_tensor = torch.tensor([item_id], dtype=torch.long, device=device)

    # 1. Get embeddings from both pathways
    user_mlp_embedding = model.embedding_user_mlp(user_idx_tensor)   # [1, latent_dim_mlp]
    item_mlp_embedding = model.embedding_item_mlp(item_idx_tensor)   # [1, latent_dim_mlp]
    user_mf_embedding  = model.embedding_user_mf(user_idx_tensor)    # [1, latent_dim_mf]
    item_mf_embedding  = model.embedding_item_mf(item_idx_tensor)    # [1, latent_dim_mf]

    # 2. Compute the MF element-wise product
    mf_product = user_mf_embedding * item_mf_embedding

    # 3. MLP forward pass: concatenate user/item MLP embeddings and pass through hidden layers
    mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
    mlp_output = mlp_input
    for layer in model.fc_layers:
        mlp_output = layer(mlp_output)
        mlp_output = torch.nn.functional.relu(mlp_output)

    # 4. Final concatenation: combine mlp_output with mf_product
    final_vector = torch.cat([mlp_output, mf_product], dim=-1)

    # 5. Final layer: compute contributions and predicted score
    affine_weight = model.affine_output.weight[0]  # [final_dim]
    affine_bias = model.affine_output.bias[0]        # scalar
    final_contributions = final_vector.view(-1) * affine_weight

    logits = model.affine_output(final_vector)
    raw_rating = model.logistic(logits).item()
    
    if model.config.get("is_explicit", False):
        predicted_score = raw_rating * model.config.get("max_rating", 1)
    else:
        predicted_score = raw_rating

    # Helper function: convert tensor to Python list
    def to_list(tensor):
        return tensor.detach().cpu().view(-1).tolist()

    # Construct the explanation dictionary
    explanation = {
        "user_id": user_id,
        "item_id": item_id,
        "user_mlp_embedding": to_list(user_mlp_embedding),
        "item_mlp_embedding": to_list(item_mlp_embedding),
        "user_mf_embedding": to_list(user_mf_embedding),
        "item_mf_embedding": to_list(item_mf_embedding),
        "mf_product": to_list(mf_product),
        "mlp_output": to_list(mlp_output),
        "final_vector": to_list(final_vector),
        "final_contributions": to_list(final_contributions),
        "raw_score": raw_rating,
        "predicted_score": predicted_score,
        "affine_bias": affine_bias.item(),
        "affine_weights": to_list(affine_weight)
    }

    return explanation

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def generate_explanations_for_user(model, user_id, top_n_items, base_dir="explanations"):
    """Generate explanation JSONs for top N items for a specific user"""
    # Convert user_id from numpy type to standard Python int if needed
    if isinstance(user_id, np.integer):
        user_id = int(user_id)
        
    # Create user-specific directory
    user_dir = os.path.join(base_dir, f"user_{user_id}")
    ensure_directory_exists(user_dir)
    
    for rank, item in enumerate(top_n_items, 1):
        # Convert item from numpy type to standard Python int if needed
        if isinstance(item, np.integer):
            item = int(item)
            
        # Run the detailed forward pass
        explanation = forward_explain_detailed(model, user_id, item)
        
        # Save the explanation to a JSON file
        output_filename = os.path.join(user_dir, f"rank_{rank}_item_{item}.json")
        with open(output_filename, "w") as f:
            json.dump(explanation, f, indent=4, cls=NumpyEncoder)
        
        # print(f"Explanation for user {user_id}, item {item} (rank {rank}) saved to {output_filename}")

if __name__ == "__main__":
    # Config
    config = {
        "num_users": 610,
        "num_items": 9724,
        "latent_dim_mf": 32,
        "latent_dim_mlp": 32,
        "layers": [64, 128, 64, 32, 16],
        "is_explicit": True,     # Set to False if using implicit feedback
        "use_cuda": True,        # Change to True if using a GPU
        "weight_init_gaussian": True  
    }

    ml1m_rating = pd.read_csv('data/ml-latest-small/ratings.csv')
    ml1m_rating.rename(columns={'userId': 'uid', 'movieId': 'mid'}, inplace=True)
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    num_users = ml1m_rating['userId'].nunique()
    num_items = ml1m_rating['itemId'].nunique()
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    
    config['num_users'] = num_users
    config['num_items'] = num_items

    sample_generator = SampleGenerator(ratings=ml1m_rating)
    config['max_rating'] = sample_generator.max_rating

    # Instantiate the model
    model = NeuMF(config)

    # Load your trained model checkpoint
    checkpoint_path = "checkpoints/explicit/run8/neumf_factor8neg4_Epoch7_RMSE0.9691_MAE0.7390.model"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if model.config.get("use_cuda", False) else "cpu")
    model.to(device)
    model.eval()

    # Create base explanations directory
    explanations_dir = "explanations"
    ensure_directory_exists(explanations_dir)

    # Load recommendations file
    df_recs = pd.read_csv('recommendations-out/run13/diverse_recs_epoch_7.csv')
    
    # Ensure we have the right columns and handle column case mismatches
    if 'user' not in df_recs.columns and 'User' in df_recs.columns:
        df_recs.rename(columns={'User': 'user'}, inplace=True)
    if 'itemId' not in df_recs.columns and 'ItemId' in df_recs.columns:
        df_recs.rename(columns={'ItemId': 'itemId'}, inplace=True)
    if 'score' not in df_recs.columns and 'Score' in df_recs.columns:
        df_recs.rename(columns={'Score': 'score'}, inplace=True)
        
    # Verify we have all needed columns
    required_cols = ['user', 'itemId', 'score']
    if not all(col in df_recs.columns for col in required_cols):
        print(f"WARNING: Missing columns in recommendations file. Available columns: {df_recs.columns.tolist()}")
        exit(1)
    
    # Keep only necessary columns
    df_recs = df_recs[required_cols]

    # Dump first few rows for debugging
    print("First 5 rows of recommendations file:")
    print(df_recs.head(5))
    
    # Get count of unique users in the recommendations
    user_ids = [52, 132, 307]
    
    # For each user, generate explanations for their top 5 items
    start_time = time.time()
    
    print(f"Generating explanations for selected users...")
    
    for i, current_user_id in enumerate(user_ids, 1):
        user_recs = df_recs[df_recs['user'] == current_user_id]
            
        # Sort by score in descending order
        user_recs = user_recs.sort_values(by='score', ascending=False)
        
        # Get top 5 items
        top_5_items = user_recs['itemId'].tolist()
        
        
        # Generate and save explanations
        generate_explanations_for_user(model, current_user_id, top_5_items)
    
    end_time = time.time()
    print(f"All explanations generated. Total runtime: {end_time - start_time:.2f} seconds")