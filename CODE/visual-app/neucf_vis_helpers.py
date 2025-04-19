import json
from vis_helpers import scale_sim, get_movie_poster, get_movie_title
import numpy as np
import os
import dash_bootstrap_components as dbc
from dash import dcc, html

def create_main_nodes(user_id=358, movie_id=232, id_iteration=0) -> dict:
    """Generates the main nodes for NeuMF, includes:
        - User  
        - Movie
        - Bias
        - Sigmoid
    """
    
    with open(f'Data/user_{user_id}_item_{movie_id}.json') as f:
        neumf_data = json.load(f)
    
    predicted_score = neumf_data["predicted_score"]
    bias = neumf_data["affine_bias"]
    raw_score = neumf_data["raw_score"]

    if predicted_score <= 1.5:
        color = '#ff003e'
    elif predicted_score <= 2.5:
        color = '#ff0060'
    elif predicted_score <= 3.5:
        color = '#d933c2'
    elif predicted_score <= 4.0:
        color = '#b84edd'
    elif predicted_score <= 4.5:
        color = '#8a63f1'
    else:
        color = '#4073ff'
        
    nodes = [
        { # User node
            'data': {'id': f'user_{user_id}', 'label': 'User 358'},
            'position': {'x': 0, 'y': 186},
            'classes': 'neumf_user top_node',
            'style': {
                'width': 70,
                'height': 70,
                "background-image": "/assets/user_1.png",
                "background-fit": "cover",
                "background-opacity": 1,
            }
        },
        { # Movie node
            'data': {'id': f'movie_{movie_id}{id_iteration}', 'label': get_movie_title(movie_id)},
            'position': {'x': 0, 'y': 642 + 50},
            'classes': 'neumf_movie top_node',
            'style': {
                'width': 70,
                'height': 70,
                'background-image': get_movie_poster(movie_id),
                'background-fit': 'contain',
                'shape': 'rectangle',
                'background-opacity': 0.3,
                'label': ""
            }
        },
        # { # Bias node
        #     'data': {'id': 'bias', 'label': 'bias'},
        #     'position': {'x': 850, 'y': 0},
        #     'classes': 'neumf_bias top_node', 
        #     'style': {
        #         'font-style': 'italic',
        #         'width': 40,
        #         'height': 40,
        #         'background-image': '/assets/letter-b.png',
        #         'background-fit': 'contain',
        #     }
        # },
        # { # Sigmoid node
        #     'data': {'id': 'sigmoid', 'label': 'sigmoid'},
        #     'position': {'x': 1000, 'y': 420},
        #     'classes': 'neumf_sigmoid top_node',
        #     'style': {
        #         'background-image': '/assets/sigma.png',
        #         'background-fit': 'contain',
        #         'width': 50,
        #         'height': 50,
        #     }
        # },
        { # Bias node
            'data': {'id': 'bias', 'label': 'bias', 'affine_bias': bias},
            'position': {'x': 850, 'y': 0},
            'classes': 'neumf_bias top_node', 
            'style': {
                'font-style': 'italic',
                'width': 40,
                'height': 40,
                'background-image': '/assets/letter-b.png',
                'background-fit': 'contain',
            }
        },
        { # Sigmoid node
            'data': {'id': 'sigmoid', 'label': 'sigmoid', 'raw_score': raw_score},
            'position': {'x': 1000, 'y': 420},
            'classes': 'neumf_sigmoid top_node',
            'style': {
                'background-image': '/assets/sigma.png',
                'background-fit': 'contain',
                'width': 50,
                'height': 50,
            }
        },
        { # Y hat (Prediction) node
            'data': {'id': f'yhat{id_iteration}', 'label': str(round(predicted_score, 2))},
            'position': {'x': 1150, 'y': 420},
            'classes': 'neumf_yhat',
            'style': {
                'width': 70,
                'height': 70,
                'background-color': color,
                'background-opacity': 0.8,
            }
        },
        { # Dot node
            'data': {'id': 'dot', 'label': ''},
            'position': {'x': 775, 'y': 420},
            'style': {
                'background-image': '/assets/dot.png',
                'background-fit': 'cover',
                'background-opacity': 0,
                'width': 30,
                'height': 30,
            }
        }
    ]
    # Parent nodes for rectangles, do NOT set positions
    parents = [
        {
            'data': {
                'id': 'user_embed', 
                'label': 'User Embedding', 
                'dimensions': 32, 
                'color': 'blue',
                'start_x': 100,
                'start_y': 0,
            },
            'classes': 'embedding'
        },
        {
            'data': {
                'id': 'item_embed', 
                'label': 'Item Embedding', 
                'dimensions': 32, 
                'color': 'blue',
                'start_x': 100,
                'start_y': 456 + 50,
            },
            'classes': 'embedding'
        },
        {
            'data': {
                'id': 'mlp_layer1', 
                'label': 'MLP Layer 1', 
                'dimensions': 128, 
                'color': 'blue',
                'start_x': 300,
                'start_y': -68,
            },
            'classes': 'mlp_layer'
        },
        {
            'data': {
                'id': 'mlp_layer2', 
                'label': 'MLP Layer 2', 
                'dimensions': 64, 
                'color': 'blue',
                'start_x': 400,
                'start_y': 60,
            },
            'classes': 'mlp_layer'
        },
        {
            'data': {
                'id': 'mlp_layer3', 
                'label': 'MLP Layer 3', 
                'dimensions': 32, 
                'color': 'blue',
                'start_x': 500,
                'start_y': 124,
            },
            'classes': 'mlp_layer'
        },
        {
            'data': {
                'id': 'gmf_layer', 
                'label': 'GMF Layer', 
                'dimensions': 32, 
                'color': 'blue',
                'start_x': 300,
                'start_y': 456 + 50,
            },
            'classes': 'gmf_layer'
        },
        {
            'data': {
                'id': 'mlp_output', 
                'label': 'MLP Output', 
                'dimensions': 16, 
                'color': 'blue',
                'start_x': 700,
                'start_y': 126,
            },
            'classes': 'output'
        },
        {
            'data': {
                'id': 'gmf_output', 
                'label': 'GMF Output', 
                'dimensions': 32, 
                'color': 'blue',
                'start_x': 700,
                'start_y': 342,
            },
            'classes': 'output'
        },
        {
            'data': {
                'id': 'affine_weights', 
                'label': 'Affine Weights', 
                'dimensions': 48, 
                'color': 'purple',
                'start_x': 850,
                'start_y': 138,
            },
            'classes': 'weights'
        },
    ]
    
    edges = [
        # user to user_embed
        {'data': {'source': nodes[0]["data"]["id"], 'target': 'user_embed'}, 'classes': 'lookup'},
        # movie to item_embed
        {'data': {'source': nodes[1]["data"]["id"], 'target': 'item_embed'}, 'classes': 'lookup'},
        
        # both embeddings to mlp_layer1
        {'data': {'source': 'user_embed', 'target': 'mlp_layer1'}, 'classes': 'forward_pass'},
        {'data': {'source': 'item_embed', 'target': 'mlp_layer1'}, 'classes': 'forward_pass'},
        
        # both embeddings to gmf_layer
        {'data': {'source': 'user_embed', 'target': 'gmf_layer'}, 'classes': 'forward_pass'},
        {'data': {'source': 'item_embed', 'target': 'gmf_layer'}, 'classes': 'forward_pass'},
        
        # Edges through MLP layers
        {'data': {'source': 'mlp_layer1', 'target': 'mlp_layer2'}, 'classes': 'forward_pass'},
        {'data': {'source': 'mlp_layer2', 'target': 'mlp_layer3'}, 'classes': 'forward_pass'},
        
        # Output edges
        {'data': {'source': 'mlp_layer3', 'target': 'mlp_output'}, 'classes': 'forward_pass'},
        {'data': {'source': 'gmf_layer', 'target': 'gmf_output'}, 'classes': 'forward_pass'},
        
        # affine_weights to sigmoid
        #{'data': {'source': 'affine_weights', 'target': 'sigmoid'}},
        # sigmoid to yhat
        {'data': {'source': 'sigmoid', 'target': f'yhat{id_iteration}'}},
        # bias to sigmoid
        {'data': {'source': 'bias', 'target': 'sigmoid'}, 'style': {'opacity': 0.3}},
    ]
    
    elements = {"nodes": nodes, "parents": parents, "edges": edges}
    
    return elements

def create_mini_circles(parents, user_id=358, item_id=232, id_iteration=0):
    """Loop through parent elements, and generate mini circles based on the dimension"""
    
    mini_circles = []
    extra_edges = []
    
    for i, parent in enumerate(parents):
        dimensions = parent["data"]["dimensions"]
        parent_id = parent["data"]["id"]
        x_pos = parent["data"]["start_x"]
        y_pos = parent["data"]["start_y"]
        # Default size and spacing
        size = 10
        y_diff = 12
        if parent.get("classes", "None") == "mlp_layer":
            size = 3
            y_diff = 4
        
        
        # load json file for mini circles values
        with open(f'Data/user_{user_id}_item_{item_id}.json') as f:
            neumf_data = json.load(f)
            
        if parent_id == "user_embed":
            vals = neumf_data["user_mlp_embedding"]
            vals_2 = neumf_data["user_mf_embedding"]
        if parent_id == "item_embed":
            vals = neumf_data["item_mlp_embedding"]
            vals_2 = neumf_data["item_mf_embedding"]
        elif parent_id == "mlp_output":
            vals = neumf_data["final_vector"]
        elif parent_id == "gmf_output" or parent_id == "gmf_layer":
            vals = neumf_data["final_vector"]
        elif parent_id == "affine_weights":
            vals = neumf_data["affine_weights"]
        elif "mlp_layer" in parent_id:
            vals = [1] * dimensions
            
        colors = scale_sim(np.abs(vals))
        colors = np.where(colors > 0.15, colors, 0.15)
        if "mlp_layer" in parent_id:
            extra_class = "mlp_layer_mini_circle"
        elif parent_id == "user_embed" or parent_id == "item_embed":
            extra_class = "embed_mini_circle"
        else:
            extra_class = "mini_circle"
            
        final_contributions = np.array(neumf_data["final_contributions"])
        scaled_contributions = scale_sim(np.abs(final_contributions))
        scaled_contributions = np.where(scaled_contributions > 0.1, scaled_contributions, 0.1)
        contribution_colors = np.where(final_contributions > 0, "#4073ff", "#ff003e")
        
        for cir_i in range(dimensions):
            if parent_id == "gmf_output" or parent_id == "gmf_layer":
                cir_i += 16
            mini_circle = {
                "data": {
                    "id": f"{parent_id}_mini{cir_i}{id_iteration}",
                    "label": "",
                    "parent": parent_id,
                    "value": vals[cir_i],
                },
                "position": {
                    "x": x_pos,
                    "y": y_pos,
                },
                "style": {
                    "width": size,
                    "height": size,
                    "opacity": colors[cir_i],
                }, 
                "classes": f'{parent_id}_mini_circle {extra_class}',
                "grabbable": False,
            }
            if parent_id == "user_embed" or parent_id == "item_embed":
                mini_circle["data"]["mf_value"] = vals_2[cir_i]
            if parent_id == "affine_weights":
                extra_edge = {
                    "data": {"source": f"{parent_id}_mini{cir_i}{id_iteration}", "target": "sigmoid"},
                    "style": {"opacity": scaled_contributions[cir_i], "line-color": contribution_colors[cir_i]}
                }
                extra_edges.append(extra_edge)
            mini_circles.append(mini_circle)
            y_pos += y_diff
    return mini_circles + extra_edges

def create_neumf_elements(user_id, item_id, id_iteration):
    """Creates the main elements for NeuMF visualization."""
    
    main_nodes_dict = create_main_nodes(user_id, item_id, id_iteration)
    parents = main_nodes_dict["parents"]
    mini_circles = create_mini_circles(parents, user_id, item_id, id_iteration)
    main_elements = main_nodes_dict["parents"] + main_nodes_dict["nodes"] + main_nodes_dict["edges"]
    return main_elements + mini_circles

def get_neumf_stylesheet(id_iteration):
    stylesheet = [
        {
            'selector': 'node',  # Base style for all nodes
            'style': {
                'label': '',
                'font-size': '20px',
                'text-margin-y': '-10px'
            }
        },
        {
            'selector': '$node > node', 
            'style': {
                'padding': '12px',
                'background-color': '#f0f0f0',
                'border-color': '#333',
                'border-width': 4,
                'label': 'data(label)'  
            }
        }, 
        {
            'selector': f'#yhat{id_iteration}',
            'style': {
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '20px',
                'font-weight': 'bold',
                'text-margin-y': '0px'
            }
        },
        {
            'selector': '.mlp_layer',
            'style': {
                'background-color': '#8a63f1',
                'background-opacity': 0.30,
                'border-color': '#8a63f1',
                'border-width': 4,
                'padding': '8px',
            }
        },
        {
            'selector': '.mlp_layer1_mini_circle',
            'style': {
                'background-color': '#8a63f1',
                'opacity': 0
            }
        },
        {
            'selector': '.mlp_layer2_mini_circle',
            'style': {
                'background-color': '#8a63f1',
                'opacity': 0
            }
        },
        {
            'selector': '.mlp_layer3_mini_circle',
            'style': {
                'background-color': '#8a63f1'
            }
        },
        {
            'selector': '.gmf_layer',
            'style': {
                'background-color': '#d933c2',
                'background-opacity': 0.25,
                'border-color': '#d933c2',
                'border-width': 4,
                'text-valign': 'bottom',
                'font-size': '20px',
                'text-margin-y': '10px',
            }
        },
        {
            'selector': '#gmf_output',
            'style': {
                'text-valign': 'bottom',
                'font-size': '20px',
                'text-margin-y': '10px',
                'background-color': '#d933c2',
                'background-opacity': 0.25,
                'border-color': '#d933c2',
                'border-width': 4,
            }
        },
        {
            'selector': '#mlp_output',
            'style': {
                'font-size': '20px',
                'text-margin-y': '-8px',
                'background-color': '#8a63f1',
                'background-opacity': 0.25,
                'border-color': '#8a63f1',
                'border-width': 4,
            }
        },
        {
            'selector': '.user_embed_mini_circle',
            'style': {
                'background-color': '#4073ff'
            }
        },
        {
            'selector': '.item_embed_mini_circle',
            'style': {
                'background-color': '#4073ff'
            }
        },
        {
            'selector': '.embedding',
            'style': {
                'background-color': '#4073ff',
                'background-opacity': 0.25,
                'border-color': '#4073ff',
                'border-width': 3
            }
        },
        {
            'selector': '.mlp_output_mini_circle',
            'style': {
                'background-color': '#8a63f1'
            }
        },
        {
            'selector': '.gmf_output_mini_circle',
            'style': {
                'background-color': '#d933c2'
            }
        },
        {
            'selector': '.gmf_layer_mini_circle',
            'style': {
                'background-color': '#d933c2'
            }
        },
        {
            'selector': '#affine_weights',
            'style': {
                'background-color': '#808080',
                'background-opacity': 0.25,
                'border-color': '#808080',
                'border-width': 4, 
                'border-opacity': 1
            }
        },
        {
            'selector': '.affine_weights_mini_circle',
            'style': {
                'background-color': '#808080'
            }
        },
    ]
    return stylesheet

def get_movie_options(user_id):
    movie_ids =  [int(file.split('_')[3].split(".")[0]) for file in os.listdir('Data') if f"user_{user_id}" in file]
    options = [{"label": get_movie_title(movie_id), "value": movie_id} for movie_id in movie_ids]
    return options

def get_neumf_sidecard(user_id):
    
    neumf_side_card = [
        dbc.CardHeader(html.H6(f"Select a movie to see details")),
        dbc.CardBody(
            dcc.Dropdown(
                id="neumf-movie-dropdown",
                options=get_movie_options(user_id),
                value=get_movie_options(user_id)[0]["value"],
                clearable=False,
                searchable=False,
                className="dropdown-theme",
                style={"minWidth": "150px"},
            )
        )
    ]
    return neumf_side_card
