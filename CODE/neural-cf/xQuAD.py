import torch
import pandas as pd
import numpy as np
import os
from diversity import (
    calculate_diversity_metrics, 
    analyze_recommendation_overlap
)

def generate_smooth_xquad_recommendations(self, user_ids, epoch_id, top_n, silent=False):
    """Generate recommendations using Smooth xQuAD method"""
    print("Generating Smooth xQuAD recommendations...")
    
    # First generate base recommendations
    base_recommendations = {}
    for user_id in user_ids:
        seen = self.user_seen_items.get(user_id, set())
        unseen = np.array(list(set(self.all_items) - seen))
        
        if len(unseen) == 0:
            continue
            
        # Generate predictions
        users_tensor = torch.tensor([user_id] * len(unseen))
        items_tensor = torch.tensor(unseen)
        
        if self.config['use_cuda']:
            users_tensor = users_tensor.cuda()
            items_tensor = items_tensor.cuda()
        
        with torch.no_grad():
            scores = self.model(users_tensor, items_tensor).cpu().numpy().flatten()
            if self.config.get('is_explicit', False):
                scores = scores * self.config['max_rating']
        
        # Store top 100 items for re-ranking
        items_scores = [(item, score) for item, score in zip(unseen, scores)]
        items_scores.sort(key=lambda x: x[1], reverse=True)
        base_recommendations[user_id] = items_scores[:500]  # Consider top N for re-ranking
    
    # Apply Smooth xQuAD
    lambda_param = self.config.get('xquad_lambda', 0.7)
    alpha = self.config.get('popularity_threshold', 0.3)
    
    # Define item categories based on popularity
    popularity_values = list(self.item_popularity.values())
    popularity_values.sort(reverse=True)
    threshold_idx = int(len(popularity_values) * 0.2)  # Top 20% are short head
    popularity_threshold = popularity_values[threshold_idx] if threshold_idx < len(popularity_values) else 0.5
    
    short_head = {item for item, pop in self.item_popularity.items() if pop > popularity_threshold}
    long_tail = {item for item, pop in self.item_popularity.items() if pop <= popularity_threshold}
    
    print(f"Short head items: {len(short_head)}, Long tail items: {len(long_tail)}")
    
    diversified_recs = {}
    
    for user, items_scores in base_recommendations.items():
        items_dict = dict(items_scores)
        scores_vals = list(items_dict.values())
        dynamic_min = min(scores_vals)
        dynamic_max = max(scores_vals)
        score_range = dynamic_max - dynamic_min if dynamic_max > dynamic_min else 1.0
        S = []  
        R = list(items_dict.keys())  
        
        # Calculate user preference for long tail based on history
        user_items = self.user_seen_items.get(user, set())
        if len(user_items) > 0:
            long_tail_ratio = len(user_items.intersection(long_tail)) / len(user_items)
        else:
            long_tail_ratio = 0.5  # Default preference if user has no history
        
        p_long_tail = long_tail_ratio
        p_short_head = 1.0 - p_long_tail
        
        while len(S) < min(top_n, len(R)):
            max_score = -float('inf')
            best_item = None
            
            for item in R:
                if item in S:
                    continue
                    
                # Base relevance - normalize to 0-1 range for explicit feedback
                original_score = items_dict.get(item, 0)
                if self.config.get('is_explicit', False):
                    original_score = items_dict.get(item, 0)
                    rel = (original_score - dynamic_min) / score_range
                else:
                    rel = original_score  # Already in 0-1 range for implicit
                
                # Calculate diversity component using smooth xQuAD
                diversity_score = 0
                
                # For each category (short head and long tail)
                for category, category_items, p_cat in [
                    ('short_head', short_head, p_short_head),
                    ('long_tail', long_tail, p_long_tail)
                ]:
                    # Item belongs to category?
                    p_item_in_cat = 1.0 if item in category_items else 0.0
                    
                    # Calculate how well category is already covered (smooth version)
                    items_from_cat_in_S = sum(1 for i in S if i in category_items)
                    
                    # Smooth coverage calculation
                    coverage_ratio = items_from_cat_in_S / max(1, len(S)) if S else 0
                    not_covered = 1.0 - coverage_ratio
                    diversity_score += p_item_in_cat * not_covered * p_cat
                
                # Combine relevance and diversity
                final_score = (1 - lambda_param) * rel + lambda_param * diversity_score
                
                if final_score > max_score:
                    max_score = final_score
                    best_item = item
            
            if best_item is not None:
                S.append(best_item)
                if best_item in R:
                    R.remove(best_item)
            else:
                break
        
        diversified_recs[user] = S
    
    # Convert to DataFrame
    results = []
    for user_id, items in diversified_recs.items():
        for rank, item_id in enumerate(items, 1):
            results.append({
                'user': user_id,
                'itemId': item_id,
                'score': next(score for item, score in base_recommendations[user_id] if item == item_id),
                'rank': rank,
                'category': 'short_head' if item_id in short_head else 'long_tail'
            })
    
    recs_df = pd.DataFrame(results)
    
    # Add movie information if available
    try:
        movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
        movies_df['movieId'] = movies_df['movieId'].astype(int)
        movies_df.rename(columns={'movieId': 'itemId'}, inplace=True)
        
        recs_df = recs_df.merge(
            movies_df[['itemId', 'title', 'genres']],
            on='itemId', 
            how='left'
        )
    except Exception as e:
        print(f"Warning: Could not load movie information: {e}")
    
    # Calculate diversity metrics
    diversity_metrics = calculate_diversity_metrics(recs_df)
    # Calculate overlap metrics
    overlap_metrics = analyze_recommendation_overlap(recs_df)
    
    if not silent:
        print("Diversity metrics:", diversity_metrics)
        print("Overlap metrics:", overlap_metrics)
    
    # Save recommendations
    directory = f"recommendations-out/run{self.config['run_number']}"
    os.makedirs(directory, exist_ok=True)
    
    filename = f"{directory}/smooth_xquad_recs_epoch_{epoch_id}.csv" if epoch_id is not None else f"{directory}/final_smooth_xquad_recs.csv"
    recs_df.to_csv(filename, index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        **diversity_metrics,
        **overlap_metrics,
        'epoch': epoch_id if epoch_id is not None else 'final'
    }, index=[0])
    
    metrics_filename = f"{directory}/diversity_metrics_epoch_{epoch_id}.csv" if epoch_id is not None else f"{directory}/final_diversity_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    
    print(f"Saved Smooth xQuAD recommendations to {filename}")
    
    return recs_df



