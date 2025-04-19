import pandas as pd
import numpy as np
from cf import CollaborativeFiltering
import pickle #type: ignore
from scipy.special import softmax
from dash import html
import dash_bootstrap_components as dbc

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
        
    Returns:
        tuple[list[dict], list[float]]: 1) Dictionary with keys: id, label, type, rating 2) list of predicted ratings
    """
    rec_nodes = recommendations[["movieId", "title"]].copy(deep=True)
    ratings = recommendations["rating"].tolist()
    rec_nodes["type"] = "movie"
    rec_nodes["movieId"] = rec_nodes["movieId"].apply(lambda x: "movie_" + str(x) + f"_{iteration}")
    rec_nodes = rec_nodes.astype(str).rename(columns={"movieId":"id", "title":"label"}).to_dict(orient="records")
    return rec_nodes, ratings


########## AFTER THIS LINE, THE FILE IS CHANGED FROM THE UBCF VISUAL, VERY SIMILAR STRUCTURE #############


def create_profile_elements_ibcf(prime_user: int, profile_limit: int, vis_type: str="ibcf", iteration:int=0, sim_elements=None) -> list[dict]:
    
    new_elements = []
    user_profile = create_user_profile(prime_user)
    profile_nodes = create_profile_nodes(user_profile, limit=profile_limit, iteration=iteration)
    
    
    sim_y_positions = [node["position"]["y"] for node in sim_elements]
    
    min_y = min(sim_y_positions)
    max_y = max(sim_y_positions)
    center_y = (min_y + max_y) / 2
    
    start_y = center_y - ((len(profile_nodes) - 1) * 65) / 2
    profile_ys = [start_y + i * 65 for i in range(len(profile_nodes))]
    
    
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


def create_neighbor_elements_ibcf(prime_user: int, cf: CollaborativeFiltering, item_elements: list[dict], neighbor_limit: int, iteration:int=0, sim_elements=None) -> list[dict]:

    seen_movies = [int(n["data"]["id"].split("_")[1]) for n in item_elements]
    user_set = set()
    for movie_id in seen_movies:
        user_set.update(cf.movie_to_users.get(movie_id, set()))
    user_set.discard(prime_user)

    sim_y_positions = [node["position"]["y"] for node in sim_elements]
    
    min_y = min(sim_y_positions)
    max_y = max(sim_y_positions)
    center_y = (min_y + max_y) / 2
    
    start_y = center_y - ((neighbor_limit - 1) * 65) / 2
    neighbor_ys = [start_y + i * 65 for i in range(neighbor_limit)]


    new_elements = []
    for idx, uid in enumerate(list(user_set)[:neighbor_limit]):
        new_elements.append({
            "data": {
                "id": f"user_{uid}_{iteration}",
                "label": f"User {uid}",
                "type": "user"
            },
            "style": {
                "background-image": "/assets/user_1.png",
                "background-fit": "cover",
                "background-opacity": 1,
                "width": 60,        
                "height": 60,
            },
            "position": {
                "x": 1000,         
                "y": neighbor_ys[idx]    
            }
        })

    return new_elements




def create_similar_elements_ibcf(seen_ids: list[dict], cf: CollaborativeFiltering, limit: int = 20, iteration:int=0) -> list[dict]:
    
    #seen_ids = [int(n["data"]["id"].split("_")[1]) for n in item_elements]
    candidates = []
    # for each movie in the profile, get the most similar movies
    for movie_id in seen_ids:
        neighbors = cf.item_neighbors.get(movie_id, [])
        for neighbor_id, sim, _ in neighbors:
            # if the neighbor is not in the profile, add it to the candidates
            if neighbor_id not in seen_ids:
                candidates.append((neighbor_id, sim))

    unique = {}
    # loop through the candidates and keep the most similar movies
    for movie_id, sim in sorted(candidates, key=lambda x: x[1], reverse=True):
        if movie_id not in unique:
            unique[movie_id] = sim
        if len(unique) >= limit:
            break

    elements = []
    for idx, (mid, sim) in enumerate(unique.items()):
        title = movies[movies["movieId"] == mid]["title"].values[0]
        elements.append({
            "data": {
                "id": f"movie_{mid}_{iteration}",
                "label": title,
                "type": "movie",
                "similarity": sim
            },
            "style": {
                "background-image": get_movie_poster(mid),
                "background-fit": "contain",
                "background-opacity": 0.8,
                "width": 50,
                "height": 50,
            },
            "position": {"x": 1600, "y": idx * 65}
        })
    return elements


def create_rec_elements_ibcf(prime_user: int, cf: CollaborativeFiltering, rec_limit: int = 5, sim_movie_elements=None, top_n=30) -> list[dict]:
    sim_ys = [node["position"]["y"] for node in sim_movie_elements]
    center_y = sum(sim_ys) / len(sim_ys) if sim_ys else 0
    start_y = center_y - ((rec_limit - 1) * 150) / 2
    recommendation_ys = [start_y + i * 150 for i in range(rec_limit)]

    recommendations = cf.generate_recommendations(prime_user, limit=rec_limit, method="ibcf", top_n=top_n)
    rec_nodes, _ = create_rec_nodes(recommendations)
    
    new_elements = []
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
            "position": {"x": 2200, "y": y}
        })
    return new_elements


def create_edge_elements_ibcf(prime_user, profile_elements, neighbor_elements, sim_movie_elements, rec_elements, cf: CollaborativeFiltering, iteration:int=0):
    edges = []

    # Profile ➝ Users
    for item in profile_elements:
        movie_id = int(item["data"]["id"].split("_")[1])
        for user in neighbor_elements:
            uid = int(user["data"]["id"].split("_")[1])
            if movie_id in cf.user_to_movies.get(uid, set()):
                rating = ratings[(ratings["userId"] == uid) & (ratings["movieId"] == movie_id)]["rating"].values[0]
                edges.append({"data": {"source": item["data"]["id"], "target": user["data"]["id"], "rating": rating}})

    # Users ➝ Similar Movies
    for sim_movie in sim_movie_elements:
        movie_id = int(sim_movie["data"]["id"].split("_")[1])
        for user in neighbor_elements:
            uid = int(user["data"]["id"].split("_")[1])
            if movie_id in cf.user_to_movies.get(uid, set()):
                rating = ratings[(ratings["userId"] == uid) & (ratings["movieId"] == movie_id)]["rating"].values[0]
                edges.append({"data": {"source": user["data"]["id"], "target": sim_movie["data"]["id"], "rating": rating}})

    # Similar Movies ➝ Recommendations
    for rec in rec_elements:
        movie_id = int(rec["data"]["id"].split("_")[1])
        for sim_movie in sim_movie_elements:
            sim_id = int(sim_movie["data"]["id"].split("_")[1])
            neighbors = cf.item_neighbors.get(sim_id, [])

            if any(n[0] == movie_id for n in neighbors):
                
                rating_row = ratings[
                    (ratings["userId"] == prime_user) & 
                    (ratings["movieId"] == movie_id)
                ]

                if not rating_row.empty:
                    rating = rating_row["rating"].values[0]
                else:
                    rating = cf.predict_item_based(prime_user, movie_id, top_n=30) or 0
                edges.append({
                    "data": {
                        "source": sim_movie["data"]["id"],
                        "target": rec["data"]["id"],
                        "rating": rating  
                    }
                })

    
    return edges

def get_ibcf_stylesheet():
    ibcf_stylesheet = [
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
    return ibcf_stylesheet



def create_ibcf_elements(prime_user:int, profile_limit:int, neighbor_limit:int, sim_limit:int, rec_limit:int, cf:CollaborativeFiltering, id_iteration:int) -> list[dict]:
    """Generate all Cytoscape elements for IBCF visualization (clean format for graph_app.py)."""
    elements = []
    if neighbor_limit <= 30:
        offscreen_x = 3000
    elif neighbor_limit <= 60:
        offscreen_x = 5000
    elif neighbor_limit <= 100:
        offscreen_x = 10000
    
    seen_ids = cf.user_to_movies.get(prime_user, set())
    
    sim_elements = create_similar_elements_ibcf(seen_ids, cf, limit=sim_limit, iteration=id_iteration)
    target_x = 1600
    for el in sim_elements:
        el["data"]["final_position"] = {"x": target_x, "y": el["position"]["y"]}
        el["position"]["x"] = target_x + offscreen_x
        el["style"]["visibility"] = "hidden"
        el["classes"] = "neighbor"

    elements.extend(sim_elements)

    profile_elements = create_profile_elements_ibcf(prime_user, profile_limit, vis_type="ibcf", iteration=id_iteration, sim_elements=sim_elements)
    target_x = 400  # where UBCF profile nodes land
    for el in profile_elements:
        el["data"]["final_position"] = {"x": target_x, "y": el["position"]["y"]}
        el["position"]["x"] = target_x - 300
        el["style"]["visibility"] = "hidden"
        el["classes"] = "prime_user"

    elements.extend(profile_elements)


    neighbor_elements = create_neighbor_elements_ibcf(prime_user, cf, profile_elements, neighbor_limit, iteration=id_iteration, sim_elements=sim_elements)
    target_x = 1000
    for el in neighbor_elements:
        el["data"]["final_position"] = {"x": target_x, "y": el["position"]["y"]}
        el["position"]["x"] = target_x + offscreen_x
        el["style"]["visibility"] = "hidden"
        el["classes"] = "profile"


    elements.extend(neighbor_elements)

   


    rec_elements = create_rec_elements_ibcf(prime_user, cf, rec_limit=rec_limit, sim_movie_elements=sim_elements, top_n=sim_limit)
    target_x = 2550
    for el in rec_elements:
        el["data"]["final_position"] = {"x": target_x, "y": el["position"]["y"]}
        el["position"]["x"] = target_x + offscreen_x
        el["style"]["visibility"] = "hidden"
        el["classes"] = "rec"

    elements.extend(rec_elements)

    
##### WASNT SURE IF YOU GUYS WANTED TO KEEP THE USER NODE #####
    neighbor_y_positions = [el["position"]["y"] for el in neighbor_elements]
    center_y = sum(neighbor_y_positions) / len(neighbor_y_positions) if neighbor_y_positions else 0
    # prime_user_element = {
    #     "data": {
    #         "id": str(prime_user),
    #         "label": f"User {prime_user}",
    #         "type": "user",
    #         "final_position": {"x": 0, "y": center_y}  # ← Add this
    #     },
    #     "style": {
    #         "background-image": "/assets/user_1.png",
    #         "background-fit": "cover",
    #         "background-opacity": 1,
    #         "width": 50,
    #         "height": 50,
    #         "visibility": "hidden"
    #     },
    #     "position": {"x": -200, "y": center_y},  # ← Start offscreen
    #     "selectable": False,
    #     "locked": False,
    #     "classes": "prime_user"
    # }

    # elements.append(prime_user_element)
    edge_groups = create_edge_elements_ibcf(
        prime_user, profile_elements, neighbor_elements, sim_elements, rec_elements, cf
    )

    for edge in edge_groups:
        edge["style"] = {"opacity": 0}
        
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        if source.startswith("movie_") and target.startswith("user_"):
            edge["classes"] = "profile_edge cf_edge"
        elif source.startswith("user_") and target.startswith("movie_"):
            edge["classes"] = "crowd_profile_edge cf_edge"
        else:
            edge["classes"] = "crowd_rec_edge cf_edge"
        
        elements.append(edge)


    return elements
