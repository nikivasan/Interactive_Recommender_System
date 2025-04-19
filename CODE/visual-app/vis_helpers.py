import pandas as pd
import numpy as np
from cf import CollaborativeFiltering
import pickle #type: ignore
from scipy.special import softmax
from dash import html
import dash_bootstrap_components as dbc

# Read in data 
ratings = pd.read_csv("Data/ml-latest-small/ratings.csv")
movies = pd.read_csv("Data/ml-latest-small/movies.csv")[["movieId", "title"]]
ratings = ratings.merge(movies, on="movieId", how="left").drop(columns="timestamp")
movie_data = pd.read_csv("Data/tmdb_movie_details.csv")


def get_movie_title(movie_id:int) -> str:
    """Get movie title for a given movieId"""
    if movie_id in ratings["movieId"].values:
        title = movies[movies["movieId"] == movie_id]["title"].values[0]
    else:
        return "Unk"
    return title

def create_user_profile(user_id:int) -> pd.DataFrame:
    """Get movies and ratings for a given user
    
    Args:
        user_id (int): user id
        
    Returns:
        pd.DataFrame: rows of movieId, rating, title
    """
    user_profile = ratings[ratings["userId"] == user_id].copy(deep=True)
    user_profile.drop(columns="userId", inplace=True)
    user_profile.reset_index(drop=True, inplace=True)
    return user_profile

def create_profile_nodes(user_profile:pd.DataFrame, limit:int=15, iteration:int=0) -> list[dict]:
    """Create nodes (formatted dictionary) for the user profile
    
    Args:
        user_profile (pd.DataFrame): profile movies
        limit (int, optional): Number of nodes to create. Defaults to 15.
        iteration (int, optional): Iteration number for unique id. Defaults to 0.
        
    Returns:
        list[dict]: Dictionary with keys: id, label, type, rating
    """
    
    profile = user_profile.copy(deep=True)
    profile["movieId"] = profile["movieId"].apply(lambda x: "movie_" + str(x) + f"_{iteration}")
    profile["type"] = "movie"
    profile.sort_values("rating", ascending=False, inplace=True)
    profile_nodes = profile[["movieId", "title", "type", "rating"]].astype(str).rename(columns={"movieId":"id", "title":"label"}).to_dict(orient="records")
    return profile_nodes[:limit]

def scale_sim(arr, type:str="min-max") -> np.ndarray:
    if type == "softmax":
        arr = softmax(arr)
        return arr
    elif type == "min-max":
        arr = np.array(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.ones_like(arr)
        return arr

def create_neighbor_nodes(neighbors:list[tuple], neighbor_type:str="user", limit:int=15, iteration:int=0) -> tuple[list[dict], list[float]]:
    """Create nodes (formatted dictionary) for users or items
    
    Args:
        neighbors (list[tuple]): List of (userId/movieId, sim, num_common) tuples, from cf model
        limit (int, optional): Number of neighbor nodes to create. Defaults to 15.
        neighbor_type (str, optional): Type of neighbor. Defaults to "user".
        iteration (int, optional): Iteration number for unique id. Defaults to 0.
        
    Returns:
        tuple[list[dict], list[float], list[int]]: Tuple of 3 lists: 1) Node dictionaries 2) similarities 3) number of common entries
    """
    if neighbor_type not in ["user", "item"]:
        raise ValueError("neighbor_type must be 'user' or 'item'")
    if neighbor_type == "user":
        neighbor_nodes = [
            {
                "id":"user_" + str(n[0]) + f"_{iteration}", 
                "label":f"User {n[0]}", 
                "type":"user",
                "similarity":n[1],
                "num_corated":n[2]
            } for n in neighbors if pd.notna(n[1])]
    elif neighbor_type == "item":
        neighbor_nodes = [{"id":"movie_" + str(n[0]) + f"_{iteration}", "label":n[0], "type":"movie"} for n in neighbors if pd.notna(n[1])]
    
    similarities = [n[1] for n in neighbors if pd.notna(n[1])]
    num_common = [n[2] for n in neighbors if pd.notna(n[1])]
    if len(similarities) >= limit:
        similarities = scale_sim(similarities[:limit], type="min-max")
    
    return neighbor_nodes[:limit], similarities, num_common[:limit]

def initialize_cf() -> CollaborativeFiltering:
    """Initialize collaborative filtering model with pre-computed neighbors
    
    Returns:
        CollaborativeFiltering: model class instance with pre-computed neighbors
    """
    cf = CollaborativeFiltering(ratings, movies)
    
    with open('user_neighbors_2.pkl', 'rb') as f:
        user_neighbors = pickle.load(f)
        cf.user_neighbors = user_neighbors
    
    with open("item_neighbors_2.pkl", "rb") as f:
        item_neighbors = pickle.load(f)
        cf.item_neighbors = item_neighbors
        
    return cf

movie_posters = movie_data[["movieId", "poster_url"]].dropna(subset="poster_url").set_index("movieId").to_dict()["poster_url"]

def get_movie_poster(movie_id:int, movie_posters=movie_posters) -> str:
    """Get movie poster URL for a given movieId"""
    poster = movie_posters.get(movie_id, "/assets/clapperboard.png")
    if poster == "https://image.tmdb.org/t/p/w200None":
        poster = "/assets/clapperboard.png"
    return poster

def create_rec_nodes(recommendations:pd.DataFrame, iteration:int=0) -> tuple[list[dict], list[float]]:
    """Create nodes (formatted dictionary) for the recommendations
    
    Args:
        recommendations (pd.DataFrame): Generated recommendations from cf.generate_recommendations()
        iteration (int, optional): Iteration number for unique id. Defaults to 0.
        
    Returns:
        tuple[list[dict], list[float]]: 1) Dictionary with keys: id, label, type, rating 2) list of predicted ratings
    """
    rec_nodes = recommendations[["movieId", "title"]].copy(deep=True)
    ratings = recommendations["rating"].tolist()
    rec_nodes["type"] = "movie"
    rec_nodes["movieId"] = rec_nodes["movieId"].apply(lambda x: "movie_" + str(x) + f"_{iteration}")
    rec_nodes = rec_nodes.astype(str).rename(columns={"movieId":"id", "title":"label"}).to_dict(orient="records")
    return rec_nodes, ratings

# CREATING FINAL ELEMENTS FOR VISUALIZATION
# Elements are nested dictionaries with keys: data, style, position, classes, etc.

def create_profile_elements(prime_user: int, profile_limit:int, vis_type:str="ubcf", iteration:int=0, neighbor_elements=None) -> list[dict]:
    """Cytoscape elements for the user profile
    
    Args:
        prime_user (int): prime user id
        profile_limit (int): maximum number of movies to display in the profile
        vis_type (str): type of visualization ("ubcf" or "ibcf")
        iteration (int): iteration number for unique id
        neighbor_elements (list[dict]): Cytoscape elements for user neighbors
        
    Returns:
        list[dict]: Cytoscape elements for the user profile
    """
    new_elements = []
    
    # Create a DataFrame of user profile movies
    user_profile = create_user_profile(prime_user)
    # Convert the user profile into nodes, with a limit on the number of nodes
    profile_nodes = create_profile_nodes(user_profile, limit=profile_limit, iteration=iteration)
    
    neighbor_y_positions = [node["position"]["y"] for node in neighbor_elements]
    
    min_y = min(neighbor_y_positions)
    max_y = max(neighbor_y_positions)
    center_y = (min_y + max_y) / 2
    
    start_y = center_y - ((len(profile_nodes) - 1) * 65) / 2
    profile_ys = [start_y + i * 65 for i in range(len(profile_nodes))]
    
    # User profile nodes
    if vis_type == "ibcf":
        movie_x = 0
    else:
        movie_x = 600
    
    for node_data, y in zip(profile_nodes, profile_ys):
        node_data["rating"] = float(node_data["rating"])
        new_elements.append({
            "data": node_data,
            "style": {
                "background-image": get_movie_poster(int(node_data["id"].split("_")[1])),
                "background-fit": "contain",
                "background-opacity": 0.8,
                "width": 50,
                "height": 50,
            },
            "position": {"x": movie_x, "y": y}
        })
        
    return new_elements

def create_neighbor_elements(prime_user:int, cf:CollaborativeFiltering, neighbor_limit:int, iteration:int=0) -> list[dict]:
    """Cytoscape elements for user neighbors
    
    Args:
        prime_user (int): prime user id
        cf (CollaborativeFiltering): CollaborativeFiltering model instance
        neighbor_limit (int): maximum number of neighbors to display
        iteration (int): iteration number for unique id
        
    Returns:
        list[dict]: Cytoscape elements for user neighbors
    """
    new_elements = []
    
    # Extract neighbors from the collaborative filtering model
    neighbors = cf.user_neighbors[prime_user]
    # Create nodes for the neighbors, with a limit on the number of nodes
    neighbor_nodes, similarities, num_common = create_neighbor_nodes(neighbors, limit=neighbor_limit, iteration=iteration)
    
    combined = list(zip(num_common, neighbor_nodes, similarities))
    combined_sort = sorted(combined, key=lambda x: x[0], reverse=True)
    num_common_sorted, neighbor_nodes_sorted, similarities_sorted = zip(*combined_sort)
    
    user_y = 0
    for node_data, sim in zip(neighbor_nodes_sorted, similarities_sorted):
        if sim < 0:
            size = 40
            offset = 100 * sim
        else:
            size = 50 * sim + 40
            offset = 1200 * sim
        new_elements.append(
            {
                "data": node_data,
                "style": {
                    "background-image": "/assets/user_1.png", #<a href="https://www.flaticon.com/free-icons/user" title="user icons">User icons created by Laura Reen - Flaticon</a>
                    "background-fit": "cover",
                    "background-opacity": 1,
                    "width": size,
                    "height": size,
                },
                "position": {"x": 2200 - offset, "y": user_y}
            }
        )
        user_y += 65
    
    return new_elements

def create_rec_elements(prime_user, cf:CollaborativeFiltering, rec_limit, top_n:int=15, method="ubcf", neighbor_elements=None, iteration:int=0):
    """Create recommendation elements for the visualization
    
    Args:
        prime_user (int): Prime user id
        cf (CollaborativeFiltering): CollaborativeFiltering model instance
        rec_limit (int): Limit on the number of recommendations
        top_n (int, optional): Number of neighbors to use for recommendations. Defaults to 15.
        method (str, optional): Model used for rating prediction. Defaults to "ubcf".
        neighbor_elements (list[dict], optional): Cytoscape neighbor elements. Defaults to None.
        iteration (int, optional): iteration number for unique id. Defaults to 0.
        
    Returns:
        _type_: _description_
    """
    neighbor_y_positions = [node["position"]["y"] for node in neighbor_elements]
    
    min_y = min(neighbor_y_positions)
    max_y = max(neighbor_y_positions)
    center_y = (min_y + max_y) / 2
    
    start_y = center_y - ((rec_limit - 1) * 150) / 2
    recommendation_ys = [start_y + i * 150 for i in range(rec_limit)]
    
    new_elements = []
    # Recommendation nodes
    recommendations = cf.generate_recommendations(prime_user, limit=rec_limit, method=method, top_n=top_n)
    rec_nodes, ratings = create_rec_nodes(recommendations, iteration=iteration)
    
    # Recommendation nodes
    for node_data, y in zip(rec_nodes, recommendation_ys):
        new_elements.append({
            "data": node_data,
            "style": {
                "background-image": get_movie_poster(int(node_data["id"].split("_")[1])),
                "background-fit": "contain",
                "background-opacity": 0.8,
                "width": 80,
                "height": 80,
            },
            "position": {"x": 2800, "y": y}
        })
    
    return new_elements

def create_edge_elements(prime_user, profile_elements, neighbor_elements, rec_elements, cf:CollaborativeFiltering, iteration:int=0):
    """Create edge elements for the visualization"""
    
    profile_edges = [
        {"data": {"source":str(prime_user) + f"_{iteration}", "target":node["data"]["id"], "rating": float(node["data"]["rating"])}} for node in profile_elements if node["data"]["id"] != f"{prime_user}"
    ]
    
    user_profile = create_user_profile(prime_user)
    profile_limit = len(profile_elements)
    profile_movies = set(user_profile.sort_values("rating", ascending=False).iloc[:profile_limit]["movieId"])
    
    crowd_profile_edges = []
    for neighbor in neighbor_elements:
        user_id = int(neighbor["data"]["id"].split("_")[1])
        crowd_profile_edges.extend([
            {
                "data": {"source":neighbor["data"]["id"], "target":f"movie_{movie_id}_{iteration}", "rating":ratings.loc[ratings["userId"] == user_id].loc[ratings["movieId"] == movie_id, "rating"].values[0]}
            } for movie_id in cf.user_to_movies[user_id] if movie_id in profile_movies])
        
    crowd_rec_edges = []
    rec_ids = [int(node["data"]["id"].split("_")[1]) for node in rec_elements]
    
    for neighbor in neighbor_elements:
        user_id = int(neighbor["data"]["id"].split("_")[1])
        crowd_rec_edges.extend([
            {
                "data": {
                    "source":neighbor["data"]["id"], 
                    "target":f"movie_{movie_id}_{iteration}", 
                    "rating":ratings.loc[ratings["userId"] == user_id].loc[ratings["movieId"] == movie_id, "rating"].values[0],
                    "similarity": neighbor["data"]["similarity"]
                }
            } for movie_id in cf.user_to_movies[user_id] if movie_id in rec_ids])
        
    return profile_edges, crowd_profile_edges, crowd_rec_edges

def get_ubcf_stylesheet():
    ubcf_stylesheet = [
        {
            "selector": 'node[type = "movie"]', 
            "style": {
                "label": "", 
                "shape": "rectangle"
            }
        },
        {
            "selector": 'node[type = "user"]', 
            "style": {
                "label": ""
            }
        },
        {
            "selector": 'edge',
            "style": {
                "opacity": 0.05,
            }
        }
    ]
    return ubcf_stylesheet

def create_ubcf_elements(prime_user, profile_limit, neighbor_limit, rec_limit, rec_top_n, cf, id_iteration):
    """Uses the collaborative filtering model and the above functions to create the elements for the visualization."""
    
    if neighbor_limit <= 30:
        offscreen_x = 3000
        add_xspacing = 0
    elif neighbor_limit <= 60:
        offscreen_x = 5000
        add_xspacing = 600
    elif neighbor_limit <= 100:
        offscreen_x = 10000
        add_xspacing = 1000
    
    # Neighbor nodes
    neighbor_elements = create_neighbor_elements(prime_user, cf, neighbor_limit=neighbor_limit, iteration=id_iteration)
    for el in neighbor_elements:
        el["style"]["visibility"] = "hidden"
        el["data"]["final_position"] = el["position"]
        el["data"]["final_position"]["x"] = el["position"]["x"] + (add_xspacing * 2)
        el["position"] = {"x": 1200 + offscreen_x, "y": el["position"]["y"]}
        el["classes"] = "neighbor"
        
    # Prime user node and user profile nodes
    profile_elements = create_profile_elements(prime_user, profile_limit=profile_limit, iteration=id_iteration, neighbor_elements=neighbor_elements)
    for el in profile_elements:
        el["style"]["visibility"] = "hidden"
        el["data"]["final_position"] = el["position"]
        el["data"]["final_position"]["x"] = el["position"]["x"] + add_xspacing
        el["position"] = {"x": el["position"]["x"] + offscreen_x, "y": el["position"]["y"]}
        el["classes"] = "profile"
    
    # Recommendation nodes
    rec_elements = create_rec_elements(prime_user, cf, rec_limit=rec_limit, top_n=rec_top_n, neighbor_elements=neighbor_elements, iteration=id_iteration)
    for el in rec_elements:
        el["style"]["visibility"] = "hidden"
        el["data"]["final_position"] = el["position"]
        el["data"]["final_position"]["x"] = el["position"]["x"] + (add_xspacing * 3)
        el["position"] = {"x": 1200 + offscreen_x, "y": el["position"]["y"]}
        el["classes"] = "rec"
    
    # Edge elements
    profile_edges, crowd_profile_edges, crowd_rec_edges = create_edge_elements(prime_user, profile_elements, neighbor_elements, rec_elements, cf, iteration=id_iteration)
    
    for edge in profile_edges:
        edge["style"] = {"opacity": 0, "width": 5}
        edge["classes"] = "profile_edge cf_edge"
    for edge in crowd_profile_edges:
        edge["style"] = {"opacity": 0, "width": 5}
        edge["classes"] = "crowd_profile_edge cf_edge"
    for edge in crowd_rec_edges:
        edge["style"] = {"opacity": 0, "width": 5}
        edge["classes"] = "crowd_rec_edge cf_edge"
    
    elements = profile_elements + neighbor_elements + rec_elements + profile_edges + crowd_profile_edges + crowd_rec_edges
    
    neighbor_y_positions = [node["position"]["y"] for node in neighbor_elements]
    min_y = min(neighbor_y_positions)
    max_y = max(neighbor_y_positions)
    center_y = (min_y + max_y) / 2
    prime_user_element = {
            "data": {
                "id": str(prime_user) + f"_{id_iteration}", 
                "label": "User " + str(prime_user) + " (Primary User)", 
                "type": "user",
                "final_position": {"x": 0, "y": center_y}
            },
            "style": {
                "background-image": "/assets/user_1.png",
                "background-fit": "cover",
                "background-opacity": 1,
                "width": 100,
                "height": 100,
                "visibility": "hidden"
            },
            "position": {"x": -200, "y": center_y},
            "selectable": False,
            "locked": False,
            "classes": "prime_user"
    }
    elements.append(prime_user_element)
    
    return elements

def create_recommendation_panel(prime_user, recommendations, model, alpha):
    """Generate HTML for the recommendation panel."""
    if model == "neumf":
        panel = [
            dbc.CardHeader(
                html.H6(f"Recommendations using Neural Collaborative Filtering")
            ),
        dbc.CardBody(
            "Use the bottom navigation bar to view the recommendations.",
            style={
                "maxHeight": "300px",
                "overflowY": "auto",
                "paddingRight": "10px",
            }
        )
        ]
    else:
        list_items = [
            html.Div(
                children=[
                    html.Div(f"#{i+1}", style={'marginRight': '10px', 'fontWeight': 'bold', 'minWidth': '30px'}),
                    html.Img(src=get_movie_poster(row["movieId"]), style={'height': '50px', 'marginRight': '10px'}),
                    html.Div(row['title'], style={'fontWeight': 'bold', "fontSize": "15px"})
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '15px'
                }
            )
            for i, (_, row) in enumerate(recommendations.iterrows())
        ]
        panel = [
            dbc.CardHeader(
                html.H6(f"Recommendations (Model Weights: {round(alpha * 100)}% UBCF, {round((1-alpha) * 100)}% IBCF)")
            ),
            dbc.CardBody(
                list_items,
                style={
                    "maxHeight": "300px",
                    "overflowY": "auto",
                    "paddingRight": "10px",
                }
            )
        ]
    return panel
