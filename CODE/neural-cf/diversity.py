import numpy as np
import pandas as pd
from collections import Counter
import torch
import random 

def calculate_item_popularity(ratings_df):
    """
    Calculate item popularity based on interaction count
    Args:
        ratings_df: Pandas DataFrame with 'itemId' column
        
    Returns:
        Dictionary mapping itemId to popularity score (0-1 normalized)
    """
    popularity = ratings_df['itemId'].value_counts().to_dict()
    max_count = max(popularity.values())
    popularity = {item: count/max_count for item, count in popularity.items()}
    return popularity

def popularity_regularization(predictions, item_popularity, alpha=0.1):
    """
    Apply popularity regularization to predictions
    
    Args:
        predictions: Original prediction scores
        item_popularity: Dictionary mapping items to popularity scores
        alpha: Regularization strength (0-1)
        
    Returns:
        Regularized predictions
    """
    items = predictions.keys() if isinstance(predictions, dict) else range(len(predictions))
    regularized = {}
    
    for item in items:
        pop = item_popularity.get(item, 0)
        if isinstance(predictions, dict):
            regularized[item] = predictions[item] * (1 - alpha * pop)
        else:
            regularized[item] = predictions[item] * (1 - alpha * pop)
    
    return regularized

def calculate_diversity_metrics(recommendations_df):
    """
    Calculate diversity metrics for recommendations
    
    Args:
        recommendations_df: DataFrame with user, itemId, and score columns
        
    Returns:
        Dictionary of diversity metrics
    """
    # User coverage - percentage of users who got at least one recommendation
    user_coverage = recommendations_df['user'].nunique() / 610
    # Item coverage - percentage of total items that appear in recommendations
    item_coverage = recommendations_df['itemId'].nunique() / 9724
    # Item distribution - how evenly items are distributed
    item_counts = Counter(recommendations_df['itemId'])
    item_entropy = 0
    total_recs = len(recommendations_df)
    
    for count in item_counts.values():
        prob = count / total_recs
        item_entropy -= prob * np.log2(prob) if prob > 0 else 0
    
    # Normalize entropy by max possible entropy (if all items appeared equally)
    max_entropy = np.log2(len(item_counts)) if len(item_counts) > 0 else 0
    normalized_entropy = item_entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate the Gini coefficient for item distribution
    # Lower Gini means more equal distribution (better diversity)
    counts = sorted(item_counts.values())
    n = len(counts)
    if n > 0:
        cumulative_counts = np.cumsum(counts)
        gini = (n + 1 - 2 * np.sum(cumulative_counts) / (n * sum(counts))) / n
    else:
        gini = 0
        
    # Count how many items appeared only once in recommendations (unique recommendations)
    unique_item_recs = sum(1 for count in item_counts.values() if count == 1)
    unique_item_percentage = unique_item_recs / len(item_counts) if len(item_counts) > 0 else 0
    
    return {
        'user_coverage': user_coverage,
        'item_coverage': item_coverage,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini,
        'unique_item_percentage': unique_item_percentage
    }

def analyze_recommendation_overlap(recommendations_df):
    """
    Analyze how much overlap exists in recommendations across users
    
    Args:
        recommendations_df: DataFrame with user and itemId columns
        
    Returns:
        Dictionary of overlap metrics
    """
    user_recs = {}
    for user, group in recommendations_df.groupby('user'):
        user_recs[user] = set(group['itemId'].tolist())
    
    users = list(user_recs.keys())
    n_users = len(users)
    
    if n_users <= 1:
        return {
            'avg_overlap_ratio': 0,
            'users_with_unique_recs': 1.0,
            'item_concentration': 0
        }
    
    total_overlap = 0
    overlap_count = 0
    
    for i in range(n_users):
        for j in range(i+1, n_users):
            if len(user_recs[users[i]]) > 0 and len(user_recs[users[j]]) > 0:
                overlap = len(user_recs[users[i]] & user_recs[users[j]])
                max_possible = min(len(user_recs[users[i]]), len(user_recs[users[j]]))
                total_overlap += overlap / max_possible
                overlap_count += 1
    
    avg_overlap = total_overlap / overlap_count if overlap_count > 0 else 0
    
    users_with_unique_recs = 0
    for i, user_i in enumerate(users):
        is_unique = True
        for j, user_j in enumerate(users):
            if i != j and len(user_recs[user_i] & user_recs[user_j]) > 0:
                is_unique = False
                break
        if is_unique:
            users_with_unique_recs += 1
    
    users_with_unique_ratio = users_with_unique_recs / n_users
    
    item_counts = Counter()
    for items in user_recs.values():
        item_counts.update(items)
    
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_percent_count = sum(count for _, count in sorted_items[:max(1, int(len(sorted_items) * 0.1))])
    total_count = sum(count for _, count in sorted_items)
    
    item_concentration = top_10_percent_count / total_count if total_count > 0 else 0
    
    return {
        'avg_overlap_ratio': avg_overlap,
        'users_with_unique_recs': users_with_unique_ratio,
        'item_concentration': item_concentration
    }

def enhanced_diverse_recommendations(model, users, item_popularity, user_seen_items, 
                                   all_items, config, pop_reg_strength=0.7, long_tail_boost=0.2, top_k=5):
    """
    Enhanced version of diverse recommendations with better personalization
    
    Args:
        model: Trained recommendation model
        users: List of user IDs
        item_popularity: Dictionary mapping items to popularity scores
        user_seen_items: Dictionary mapping users to sets of seen items
        all_items: List of all available items
        config: Model configuration
        pop_reg_strength: Strength of popularity regularization (0-1)
        long_tail_boost: Additional boost for long-tail items (0-1)
        top_k: Number of items to recommend per user
        
    Returns:
        DataFrame with diverse recommendations
    """
    model.eval()
    results = []
    
    # Calculate popularity threshold (e.g., 80% of ratings in short head)
    pop_threshold = sorted(item_popularity.values(), reverse=True)[int(len(item_popularity) * 0.2)]
    
    for user_id in users:
        # Get user's preference for long-tail items
        seen = user_seen_items.get(user_id, set())
        if len(seen) > 0:
            seen_pop = [item_popularity.get(item, 0) for item in seen]
            user_avg_pop = sum(seen_pop) / len(seen_pop)
            # Higher values for users who prefer popular items, lower for those who like niche items
            personalized_strength = max(0.1, min(0.9, pop_reg_strength * (1 - user_avg_pop)))
        else:
            personalized_strength = pop_reg_strength
            
        unseen = np.array(list(set(all_items) - seen))
        if len(unseen) == 0:
            continue
        
        # Generate predictions
        users_tensor = torch.tensor([user_id] * len(unseen))
        items_tensor = torch.tensor(unseen)
        
        if config['use_cuda']:
            users_tensor = users_tensor.cuda()
            items_tensor = items_tensor.cuda()
        
        with torch.no_grad():
            scores = model(users_tensor, items_tensor).cpu().numpy().flatten()
            if config.get('is_explicit', False):
                scores = scores * config['max_rating']
        
        # Create scores dictionary
        scores_dict = {item: score for item, score in zip(unseen, scores)}
        
        # Apply personalized popularity penalty and long-tail boost
        adjusted_scores = {}
        for item, score in scores_dict.items():
            item_pop = item_popularity.get(item, 0)
            # Apply stronger penalty to popular items
            popularity_penalty = personalized_strength * item_pop
            # Add boost for long-tail items
            long_tail_bonus = 0
            if item_pop < pop_threshold:
                # The less popular, the more bonus
                long_tail_bonus = long_tail_boost * (1 - item_pop)
                
            adjusted_scores[item] = score * (1 - popularity_penalty) + long_tail_bonus
        
        # Select final items
        top_items = sorted(adjusted_scores.keys(), key=lambda x: adjusted_scores[x], reverse=True)[:top_k]
        
        # Add to results
        for rank, item_id in enumerate(top_items, 1):
            results.append({
                'user': user_id,
                'itemId': item_id,
                'score': scores_dict[item_id],
                'adjusted_score': adjusted_scores[item_id],
                'rank': rank
            })
    
    return pd.DataFrame(results)

def generate_diverse_recommendations_by_popularity(model, users, item_popularity, user_seen_items, 
                                    all_items, config, pop_reg_strength=0.2, top_k=5):
    """
    Generate diversity-enhanced recommendations for users with popularity regularization
    
    Args:
        model: Trained recommendation model
        users: List of user IDs
        item_popularity: Dictionary mapping items to popularity scores
        user_seen_items: Dictionary mapping users to sets of seen items
        all_items: List of all available items
        config: Model configuration
        pop_reg_strength: Strength of popularity regularization (0-1)
        top_k: Number of items to recommend per user
        
    Returns:
        DataFrame with diverse recommendations
    """
    model.eval()
    results = []
    
    for user_id in users:
        seen = user_seen_items.get(user_id, set())
        unseen = np.array(list(set(all_items) - seen))
        
        if len(unseen) == 0:
            continue
        
        users_tensor = torch.tensor([user_id] * len(unseen))
        items_tensor = torch.tensor(unseen)
        
        if config['use_cuda']:
            users_tensor = users_tensor.cuda()
            items_tensor = items_tensor.cuda()
        
        with torch.no_grad():
            scores = model(users_tensor, items_tensor).cpu().numpy().flatten()
            if config.get('is_explicit', False):
                scores = scores * config['max_rating']
        
        scores_dict = {item: score for item, score in zip(unseen, scores)}
        reg_scores = popularity_regularization(scores_dict, item_popularity, alpha=pop_reg_strength)
        top_items = sorted(reg_scores.keys(), key=lambda x: reg_scores[x], reverse=True)[:top_k]
        
        for rank, item_id in enumerate(top_items, 1):
            results.append({
                'user': user_id,
                'itemId': item_id,
                'score': scores_dict[item_id],
                'reg_score': reg_scores[item_id],
                'rank': rank
            })
    
    return pd.DataFrame(results)

def force_diversity_recommendations(model, users, item_popularity, user_seen_items, 
                                   all_items, config, pop_reg_strength=0.7, top_k=5):
    """
    Generate highly diverse recommendations using a combined approach of 
    strong popularity regularization and randomized selection
    
    Args:
        model: Trained recommendation model
        users: List of user IDs
        item_popularity: Dictionary mapping items to popularity scores
        user_seen_items: Dictionary mapping users to sets of seen items
        all_items: List of all available items
        config: Model configuration
        pop_reg_strength: Strength of popularity regularization (0-1)
        top_k: Number of items to recommend per user
        
    Returns:
        DataFrame with diverse recommendations
    """
    model.eval()
    results = []
    
    for user_id in users:
        seen = user_seen_items.get(user_id, set())
        unseen = np.array(list(set(all_items) - seen))
        
        if len(unseen) == 0:
            continue
        
        users_tensor = torch.tensor([user_id] * len(unseen))
        items_tensor = torch.tensor(unseen)
        
        if config['use_cuda']:
            users_tensor = users_tensor.cuda()
            items_tensor = items_tensor.cuda()
        
        with torch.no_grad():
            scores = model(users_tensor, items_tensor).cpu().numpy().flatten()
            if config.get('is_explicit', False):
                scores = scores * config['max_rating']
        
        scores_dict = {item: score for item, score in zip(unseen, scores)}
        
        pop_scores = {item: score * (1 - pop_reg_strength * item_popularity.get(item, 0)) 
                      for item, score in scores_dict.items()}
        
        candidate_items = sorted(pop_scores.keys(), key=lambda x: pop_scores[x], reverse=True)[:50]
        
        final_items = []
        if candidate_items:
            final_items.append(candidate_items[0])
            candidate_items.remove(candidate_items[0])
        
        while len(final_items) < top_k and candidate_items:
            selection_pool = candidate_items[:min(10, len(candidate_items))]
            next_item = random.choice(selection_pool)
            final_items.append(next_item)
            candidate_items.remove(next_item)
        
        for rank, item_id in enumerate(final_items, 1):
            results.append({
                'user': user_id,
                'itemId': item_id,
                'score': scores_dict[item_id],
                'reg_score': pop_scores[item_id],
                'rank': rank
            })
    
    return pd.DataFrame(results)
