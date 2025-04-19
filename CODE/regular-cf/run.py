#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from cf import CollaborativeFiltering

def main():
    """Main function to run the collaborative filtering recommendation system"""
    parser = argparse.ArgumentParser(description="Collaborative Filtering Recommendation System")
    parser.add_argument("--data_dir", default="data", help="Directory containing ratings.csv and movies.csv")
    parser.add_argument("--user_id", type=int, default=None, help="User ID to generate recommendations for")
    parser.add_argument("--method", choices=["ubcf", "ibcf", "hybrid"], default="hybrid", 
                      help="Method for generating recommendations")
    parser.add_argument("--num_recs", type=int, default=10, help="Number of recommendations to generate")
    parser.add_argument("--load_neighbors", action="store_true", 
                      help="Load pre-computed neighborhoods if available")
    parser.add_argument("--save_neighbors", action="store_true", 
                      help="Save computed neighborhoods for future use")
    parser.add_argument("--top_n", type=int, default=100,
                      help="Number of neighbors to consider for recommendations")
    parser.add_argument("--skip_fit", action="store_true", 
                   help="Skip fitting the model, use default alpha value")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Load ratings and movies data
    ratings_file = os.path.join(args.data_dir, "ratings.csv")
    movies_file = os.path.join(args.data_dir, "movies.csv")
    
    if not os.path.exists(ratings_file) or not os.path.exists(movies_file):
        print(f"Error: Could not find required data files in {args.data_dir}")
        print(f"Please ensure {ratings_file} and {movies_file} exist.")
        sys.exit(1)
    
    print("Loading data...")
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    
    print(f"Loaded {len(ratings)} ratings for {ratings['userId'].nunique()} users and {ratings['movieId'].nunique()} movies.")
    
    # Initialize model
    print("Initializing collaborative filtering model...")
    cf = CollaborativeFiltering(ratings, movies)
    
    # Load neighborhoods if requested and available
    user_neighbors_file = os.path.join(args.data_dir, "user_neighbors.pkl")
    item_neighbors_file = os.path.join(args.data_dir, "item_neighbors.pkl")
    
    if args.load_neighbors and os.path.exists(user_neighbors_file) and os.path.exists(item_neighbors_file):
        print("Loading pre-computed neighborhoods...")
        try:
            with open(user_neighbors_file, "rb") as f:
                cf.user_neighbors = pickle.load(f)
            with open(item_neighbors_file, "rb") as f:
                cf.item_neighbors = pickle.load(f)
            print("Neighborhoods loaded successfully!")
        except Exception as e:
            print(f"Error loading neighborhoods: {e}")
            print("Computing neighborhoods instead...")
            cf.set_neighborhoods()
    else:
        print("Computing neighborhoods (this may take a while)...")
        cf.set_neighborhoods()
    
    # Fit the model to optimize alpha for hybrid recommendations
    if not args.skip_fit:
        print("Fitting model to optimize blending weight...")
        cf.fit()
        print(f"Optimal blending weight (alpha): {cf.alpha:.4f}")
    else:
        print("Skipping model fitting (using default alpha value)")
    
    # Save neighborhoods if requested
    if args.save_neighbors:
        print("Saving computed neighborhoods...")
        with open(user_neighbors_file, "wb") as f:
            pickle.dump(cf.user_neighbors, f)
        with open(item_neighbors_file, "wb") as f:
            pickle.dump(cf.item_neighbors, f)
        print(f"Neighborhoods saved to {args.data_dir}")
    
    # If no user ID is provided, select a random user
    if args.user_id is None:
        all_users = cf.list_all_users()
        if len(all_users) > 0:
            args.user_id = np.random.choice(all_users)
            print(f"No user ID provided. Using random user: {args.user_id}")
        else:
            print("Error: No users found in the dataset.")
            sys.exit(1)
    
    # Generate recommendations
    print(f"Generating {args.num_recs} recommendations for user {args.user_id} using {args.method} method...")
    try:
        recommendations = cf.generate_recommendations(
            user_id=args.user_id,
            limit=args.num_recs,
            method=args.method,
            top_n=args.top_n
        )
        
        print("\nRecommended movies:")
        print(recommendations.to_string(index=False))
        
        # Show some movies the user has already rated
        user_ratings = ratings[ratings["userId"] == args.user_id].merge(movies, on="movieId")
        user_ratings = user_ratings.sort_values("rating", ascending=False)[["title", "rating"]]
        print(f"\nMovies user {args.user_id} has already rated (top 5):")
        print(user_ratings.head(5).to_string(index=False))
        
    except KeyError:
        print(f"Error: User {args.user_id} not found in the dataset.")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        
if __name__ == "__main__":
    main()