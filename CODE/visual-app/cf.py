import pandas as pd
import numpy as np
from collections import defaultdict
import heapq #type: ignore
import pickle #type: ignore


class CollaborativeFiltering:
    
    def __init__(self, ratings:pd.DataFrame, movies:pd.DataFrame, K=100) -> None:
        """Initialize collaborative filtering model
        
        Args:
            K (int, optional): Number of neighbors to consider for similarity. Defaults to 100.
        """
        
        self.K = K
        self.user_neighbors = {} # user to list of (neighbor, similarity, num_common) tuples
        self.item_neighbors = {} # item to list of (neighbor, similarity, num_common) tuples
        self.user_to_movies = {} # mapping of users to movies they have rated
        self.movie_to_users = {} # mapping of movies to users who have rated them
        self.ratings = ratings # original ratings data
        self.movies = movies # original movies data
        self.movie_rating_counts = ratings["movieId"].value_counts()
        self.alpha = 0.5 # default blending weight
        self.user_means = self.compute_user_mean()
        self.user_item_matrix = self.create_matrix(ratings)
        self.compute_mappings() # create mappings from user to movies and vice versa
    
    def compute_user_mean(self) -> pd.Series:
        """Compute mean rating for each user"""
        return self.ratings.groupby("userId")["rating"].mean()
    
    def create_matrix(self, data:pd.DataFrame, mean_center:bool=True) -> pd.DataFrame:
        """Create mean centered user-item matrix"""
        # Pivot to wide format by putting movieId in columns and userId in rows
        matrix = data.pivot(index="userId", columns="movieId", values="rating")
        if not mean_center:
            return matrix
        for user in matrix.index:
            # mean center the ratings for each user
            user_mean = self.user_means[user]
            matrix.loc[user] = matrix.loc[user] - user_mean
        return matrix
    
    def compute_mappings(self) -> None:
        """Create mappings (dict) from user to movies and vice versa"""
        self.user_to_movies = self.ratings.groupby("userId")["movieId"].apply(set).to_dict()
        self.movie_to_users = self.ratings.groupby("movieId")["userId"].apply(set).to_dict()
    
    def similarity(self, vec1:pd.Series, vec2:pd.Series, min_common:int=5) -> tuple[float, int]:
        """Compute cosine similarity based on common elements between two vectors
        
        Args:
            vec1 (pd.Series): First vector (ratings)
            vec2 (pd.Series): Second vector (ratings)
            min_common (int, optional): Minimum number of common elements to to considered a candidate. Defaults to 5.
            
        Returns:
            tuple[float, int]: Cosine similarity value, Number of common elements
        """
        # Create boolean mask for common elements
        common = vec1.notna() & vec2.notna()
        num_common = common.sum()
        if num_common < min_common:
            return np.nan, int(num_common)
        # calculate similarity on common elements
        vec1_common = vec1[common]
        vec2_common = vec2[common]
        dot_product = np.dot(vec1_common, vec2_common)
        norm1 = np.linalg.norm(vec1_common)
        norm2 = np.linalg.norm(vec2_common)
        # prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return np.nan, int(num_common)
        similarity = dot_product / (norm1 * norm2)
        return round(float(similarity), 5), int(num_common)
    
    def get_neighbors(self, matrix:pd.DataFrame, row_to_columns:dict, column_to_rows:dict, include_nan=True) -> defaultdict:
        """Find similar vectors (neighbors) for each row in the matrix
        
        Args:
            matrix (pd.DataFrame): user-item matrix
            row_to_columns (dict): mapping of rows to columns
            column_to_rows (dict): mapping of columns to rows
            
        Returns:
            defaultdict: mapping of rows to list of (neighbor, similarity, num_common) tuples
        """
        
        neighbors = defaultdict(list)
        
        for i in matrix.index:
            if i not in row_to_columns:
                continue
            
            # Find candidate neighbors 
            candidates = set().union(*[column_to_rows[col] for col in row_to_columns[i]]) - {i}
            
            heap = []
            for candidate in candidates:
                if candidate in matrix.index:
                    sim, num_common = self.similarity(matrix.loc[i], matrix.loc[candidate], min_common=5)
                    heap.append((candidate, sim, num_common))
            
            valid_values = [x for x in heap if not np.isnan(x[1])]
            nan_values = [x for x in heap if np.isnan(x[1])]
            # Sort by similarity and place NaN values at the end
            if include_nan:
                sorted_heap = sorted(valid_values, key=lambda x: x[1], reverse=True) + nan_values
            else:
                sorted_heap = sorted(valid_values, key=lambda x: x[1], reverse=True)
            neighbors[i] = sorted_heap
            
        return neighbors
    
    def predict_user_based(self, user_id:int, item_id:int, top_n:int=None, min_common:int=None) -> float:
        """Make prediction for single user-item pair based on user-based collaborative filtering
        
        Args:
            user_id (int): user id
            item_id (int): movie id
            top_n (int): number of neighbors to consider
            min_common (int): minimum number of common ratings for neighbors
            
        Returns:
            float: predicted rating
        """
        
        if user_id not in self.user_neighbors or item_id not in self.user_item_matrix.columns:
            return None
        
        if top_n:
            neighbors = self.user_neighbors[user_id][:top_n]
        else:
            neighbors = self.user_neighbors[user_id]
        
        sim_sum = 0
        weighted_sum = 0
        
        for neighbor, sim, num_common in neighbors:
            if pd.isna(sim):
                continue
            if min_common and num_common < min_common:
                continue
            rating = self.user_item_matrix.at[neighbor, item_id]
            if pd.notna(rating):
                sim_sum += abs(sim)
                weighted_sum += sim * rating
        if sim_sum != 0:
            prediction_centered = weighted_sum / sim_sum
            prediction = prediction_centered + self.user_means[user_id]
            return prediction
        
        return None
    
    def predict_item_based(self, user_id:int, item_id:int, top_n:int, min_common:int=None) -> float:
        """Make prediction for single user-item pair based on item-based collaborative filtering
        
        Args:
            user_id (int): user id
            item_id (int): movie id
            top_n (int): number of neighbors to consider
            
        Returns:
            float: predicted rating
        """
        
        if item_id not in self.item_neighbors or user_id not in self.user_item_matrix.index:
            return None
        
        if top_n:
            neighbors = self.item_neighbors[item_id][:top_n]
        else:
            neighbors = self.item_neighbors[item_id]
        
        sim_sum = 0
        weighted_sum = 0
        
        for neighbor, sim, num_common in neighbors:
            if pd.isna(sim):
                continue
            if min_common and num_common < min_common:
                continue
            rating = self.user_item_matrix.at[user_id, neighbor]
            if pd.notna(rating):
                sim_sum += abs(sim)
                weighted_sum += sim * rating
        if sim_sum > 0:
            prediction_centered = weighted_sum / sim_sum
            prediction = prediction_centered + self.user_means[user_id]
            return prediction
        return None
    
    def predict(self, user_id:int, item_id:int, top_n:int, alpha=None) -> float:
        """Predict recommendation using a blend of user-based and item-based collaborative filtering"""
        if not alpha:
            alpha = self.alpha
        
        ub_pred = self.predict_user_based(user_id, item_id, top_n=top_n)
        ib_pred = self.predict_item_based(user_id, item_id, top_n=top_n)
        if ub_pred is not None and ib_pred is not None:
            return alpha * ub_pred + (1 - alpha) * ib_pred 
        elif ub_pred is not None:
            return ub_pred
        elif ib_pred is not None:
            return ib_pred
        else:
            return None
            # or could fallback to global mean if no prediction possible
    
    def set_neighborhoods(self, user_only=False):
        """Compute neighborhoods for both user-based and item-based collaborative filtering"""
        # Compute neighbors
        self.user_neighbors = self.get_neighbors(
            self.user_item_matrix, 
            self.user_to_movies,
            self.movie_to_users
        )
        print("User neighbors computed")
        if user_only:
            return None
        item_matrix = self.user_item_matrix.T
        self.item_neighbors = self.get_neighbors(
            item_matrix, 
            self.movie_to_users,
            self.user_to_movies,
            include_nan=False
        )
        print("Item neighbors computed")
    
    def fit(self):
        """Optimize blending weight (alpha)"""
        #self.set_neighborhoods()
        sum1 = 0
        sum2 = 0
        for _, row in self.ratings.iterrows():
            user_id = row["userId"]
            item_id = row["movieId"]
            rating = row["rating"] # the actual rating
            ub_pred = self.predict_user_based(user_id, item_id, top_n=100)
            ib_pred = self.predict_item_based(user_id, item_id, top_n=100)
            if ub_pred is not None and ib_pred is not None:
                y = rating - ib_pred
                x = ub_pred - ib_pred
                sum1 += x * y
                sum2 += x * x
        if sum2 > 0:
            self.alpha = sum1 / sum2
        else:
            self.alpha = 0.5
    
    def evaluate(self, test_df: pd.DataFrame, method='hybrid') -> dict:
        """Evaluate model performance on test data, using RMSE and MAE
        
        Args:
            test_df (pd.DataFrame): ratings data
            
        Returns:
            dict: metrics, prediction results
        """
        results = {
            "user_ids": [],
            "movie_ids": [],
            "predictions": [],
            "actuals": [],
        }
        for _, row in test_df.iterrows():
            user_id = row["userId"]
            item_id = row["movieId"]
            actual_rating = row["rating"]
            if method == 'ubcf':
                predicted_rating = self.predict_user_based(user_id, item_id, top_n=100)
            elif method == 'ibcf':
                predicted_rating = self.predict_item_based(user_id, item_id, top_n=100)
            else:
                predicted_rating = self.predict(user_id, item_id)
            if predicted_rating is not None:
                results["predictions"].append(predicted_rating)
                results["actuals"].append(actual_rating)
                results["user_ids"].append(user_id)
                results["movie_ids"].append(item_id)
        
        if len(results["predictions"]) == 0:
            return None
        
        predictions = np.array(results["predictions"])
        actuals = np.array(results["actuals"])
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        metrics = {'RMSE': rmse, 'MAE': mae}
        results = pd.DataFrame(results)
        return metrics, results
    
    def list_all_users(self) -> list:
        return list(self.user_neighbors.keys())
    
    def list_all_items(self) -> list:
        return list(self.item_neighbors.keys())
    
    def get_corated(self, user1:int, user2:int) -> pd.DataFrame:
        """Get movies rated by both users"""
        vec1 = self.user_item_matrix.loc[user1]
        vec2 = self.user_item_matrix.loc[user2]
        common = vec1.notna() & vec2.notna()
        ratings1 = self.ratings[(self.ratings["userId"] == user1) & (self.ratings["movieId"].isin(vec1[common].index))].drop(columns=["userId"])
        ratings2 = self.ratings[(self.ratings["userId"] == user2) & (self.ratings["movieId"].isin(vec2[common].index))].drop(columns=["userId"])
        corated = pd.merge(ratings1, ratings2, on=["movieId", "title"], suffixes=(f"_{user1}", f"_{user2}"))
        return corated
    
    def generate_recommendations(self, user_id:int, limit:int=15, method='hybrid', top_n=10, min_seen=50, alpha=None) -> list[tuple]:
        """Generate recommendations for a given user
        
        Args:
            user_id (int): id of user
            limit (int, optional): Number of recommendations to generate. Defaults to 15.
            method (str, optional): Which algorithm to use for recommendations. Defaults to 'hybrid'.
            top_n (int, optional): Number of neighbors to use for recommendations. Defaults to 10.
            min_seen (int, optional): Minimum number of ratings to be a candidate movie. Defaults to 30.
            
        Raises:
            ValueError: If method is not one of 'ubcf', 'ibcf', or 'hybrid'
            
        Returns:
            list[tuple]: List of (movie_id, rating) tuples
        """
        recommendations = []
        
        user_profile = self.user_to_movies[user_id]
        
        neighbor_movies = set()
        if method == "ubcf":
            # If UBCF, get candidate movies from most similar users
            for neighbor_id, _, _ in self.user_neighbors[user_id][:top_n]:
                neighbor_movies.update(self.user_to_movies[neighbor_id])
        elif method == "ibcf":
            # If IBCF, get candidate movies from most similar movies to user profile
            for movie_id in user_profile:
                if movie_id in self.item_neighbors:
                    neighbor_movies.update([x[0] for x in self.item_neighbors[movie_id][:top_n]])
        elif method == "hybrid":
            # If hybrid, use both methods
            for movie_id in user_profile:
                if movie_id in self.item_neighbors:
                    neighbor_movies.update([x[0] for x in self.item_neighbors[movie_id][:top_n]])
            # Add user-based neighbors
            for neighbor_id, _, _ in self.user_neighbors[user_id][:top_n]:
                neighbor_movies.update(self.user_to_movies[neighbor_id])
        
        # Get candidate movies
        candidate_movies = list(neighbor_movies - set(user_profile))
            
        for movie_id in candidate_movies:
            if self.movie_rating_counts.loc[movie_id] < min_seen:
                continue
            if method == 'ubcf':
                rating = self.predict_user_based(user_id, movie_id, top_n)
            elif method == 'ibcf':
                rating = self.predict_item_based(user_id, movie_id, top_n)
            elif method == 'hybrid':
                rating = self.predict(user_id, movie_id, top_n, alpha=alpha)
            else:
                raise ValueError("Invalid method")
            if rating is not None:
                recommendations.append((movie_id, rating))
        top_recs = sorted(recommendations, key=lambda x: x[1], reverse=True)[:limit]
        top_recs = pd.DataFrame(top_recs, columns=["movieId", "rating"]).merge(self.movies, on="movieId", how="left")[["movieId", "title", "rating"]]
        return top_recs