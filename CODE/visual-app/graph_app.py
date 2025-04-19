from dash import Dash, html, dcc, Input, Output, callback, clientside_callback, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_ag_grid as dag


from vis_helpers import ( 
    initialize_cf, 
    get_movie_poster, 
    create_recommendation_panel, 
    get_ubcf_stylesheet,
    create_ubcf_elements,
)
# IBCF helpers
from ibcf_vis_helpers import (
    create_ibcf_elements,
    get_ibcf_stylesheet
)

from neucf_vis_helpers import (
    create_neumf_elements,
    get_neumf_stylesheet,
    get_movie_options,
)

app = Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css",
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
])

cf = initialize_cf()
all_users = cf.list_all_users()
all_movies = cf.list_all_items()

# ** COMPONENTS **
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col( 
                    html.Img(src="/assets/moviemapicon.png", height="30px"), #<a href="https://www.flaticon.com/free-icons/pin" title="pin icons">Pin icons created by berkahicon - Flaticon</a>
                    width="auto"
                ),
                dbc.Col(
                    dbc.NavbarBrand("MapMyMovies"),
                    width="auto"
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                                html.I(className="bi bi-question-circle-fill", id="about-icon"),
                                id="about-button",
                                color="link",
                                size="sm",
                                n_clicks=0,
                                style={"padding": "0", "margin": "0"}
                            ),
                            dbc.Switch(
                                id="theme-switch", 
                                label="Dark Mode",
                                label_style={"color": "white"},
                                value=False, 
                                className="form-check form-switch custom-switch",
                                style={"marginBottom": "0", "marginTop": "0.5rem", "marginLeft": "0.75rem"}
                            )
                        ],
                        className="d-flex align-items-center"
                    ),
                    width="auto",
                    className="ms-auto"
                ),
            ],
            align="center",
            className="w-100",
            justify="between"
        ),
        fluid=True,
    ),
    sticky="top"
)

# modal = dbc.Modal(
#     [
#         dbc.ModalHeader(dbc.ModalTitle("Welcome to MapMyMovies!")),
#         dbc.ModalBody([
#             html.H5("What are you looking at?"),
#             html.P("The nodes in the graph represent either users and movies (for user-based or item-based collaborative filtering), or the internal structure of a neural network (for NeuMF)."),
#             html.H5("How do you navigate the graph?"),
#             html.Ul([
#                 html.Li("Scroll to zoom in/out."),
#                 html.Li("Click and drag to pan around the graph."),
#                 html.Li("Hover over nodes to view details like user ID, movie title, network activations, etc."),
#             ]),
#             html.H5("User-based Collaborative Filtering"),
#             html.P("User-based collaborative filtering identifies patterns among users with similar tastes and preferences to generate personalized recommendations. In our visualization, the far left node represents a given user. This user is connected to a series of movies they have interacted with, or their “user profile.” Other users in the dataset are visualized in the space to the right of the user profile."),
#             html.P("Additional Interactivity:", style={'fontWeight': 'bold'}),
#             html.Ul([
#                 html.Li("Edges represent a user’s rating of a given movie."),
#                 html.Li("Edge coloring is on a scale from red to blue, where redder lines represent ratings closer to 0 and bluer lines represent ratings closer to 5."),
#                 html.Li("Similar neighbors are both larger and closer to the primary user."),
#                 html.Li("More information like user ratings or similarity scores are available upon hover."),
#             ]),
#             html.H5("Item-based Collaborative Filtering"),
#             html.P("This graph illustrates how your movie recommendations are generated using item-based collaborative filtering. Starting from the left, you’ll see the movies you’ve rated, followed by users who have also rated those same movies. Next are movies that are similar to the ones in your profile, based on shared rating patterns. The far right layer contains your personalized recommendations. The connecting edges show who rated what, and their colors reflect the strength of each rating, allowing you to trace how your preferences connect to similar content through other users. Edges from the 3rd layer to the recommendations show how the similarity-weighted ratings contribute to the final predicted rating. The strength of this connection influences which movies are promoted to the recommended list."),
#             html.H5("Hybrid Model"),
#             html.P("The hybrid model, which is a weighted blend of both user-based and item-based collaborative filtering, is captured using our visual as well. The recommendation panel on the right shows the recommended movies based on your choice of weighting between the two model outputs (e.g. 50-50, 30-70). This weighting, as well as graph size and number of visualized recommendations, can be changed using the additional settings panel."),
#             html.P("While the graphs themselves show individual model recommendations, if you weight UBCF as 100%, the graph recommendations for UBCF would be identical to the recommendation panel. Conversely, if you weight UBCF as 0, the graph recommendations for IBCF would be identical to the recommendation panel."),
#             html.H5("Neural Matrix Factorization (NeuMF)"),
#             html.P("Neural Matrix Factorization combines traditional matrix factorization with deep neural networks, visualized as user and item embeddings flowing through network layers and combined through element-wise matrix multiplication. The goal of this visualization is to demonstrate how a model selects a particular movie to be recommended to a given user. On the bottom navigation bar, you can select the user_id and movie name to explore."),
#             html.H5("Why this is useful"),
#             html.P("These visualizations help you understand why a recommendation was made — for instance, because similar users liked the same movie, or because a neural network connected your profile strongly to certain features."),
#         ]),
#         dbc.ModalFooter(
#             dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
#         ),
#     ],
#     id="modal",
#     is_open=False,
#     size="lg"
# )
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Welcome to MapMyMovies!")),
        dbc.ModalBody([
            html.H4("MapMyMovies Guide", className="mb-3"),
            
            html.H5("Overview"),
            html.P("This interactive visualization tool demonstrates how different recommendation algorithms work to suggest movies you might enjoy."),
            
            html.H5("Navigation Basics"),
            html.Ul([
                html.Li("Zoom: Use mouse scroll wheel to zoom in and out"),
                html.Li("Pan: Click and drag to move around the graph"),
                html.Li("Details: Hover over any node or connection to view additional information"),
            ]),
            
            html.H5("Understanding the Models"),
            
            html.H6("User-Based Collaborative Filtering", className="mt-3"),
            html.P("This model finds users similar to you and recommends what they enjoyed."),
            
            html.P("What you see:", className="mb-1 fw-bold"),
            html.Ul([
                html.Li("Left side: Your user profile"),
                html.Li("Middle: Profile movies you've rated"),
                html.Li("Right side: Similar users with their ratings"),
                html.Li("Far right: Recommended movies"),
            ], className="mb-2"),
            
            html.P("Interactive features:", className="mb-1 fw-bold"),
            html.Ul([
                html.Li("Color-coded connections: Red (low ratings) to blue (high ratings)"),
                html.Li("Similar users appear larger and closer to your profile"),
                html.Li("Hover over connections to see exact ratings and similarity scores"),
            ]),
            
            html.H6("Item-Based Collaborative Filtering", className="mt-3"),
            html.P("This model recommends movies similar to ones you've already enjoyed."),
            
            html.P("What you see:", className="mb-1 fw-bold"),
            html.Ul([
                html.Li("Left side: Movies you've rated"),
                html.Li("Middle-left: Users who also rated those movies"),
                html.Li("Middle-right: Similar movies based on rating patterns"),
                html.Li("Right side: Your personalized recommendations"),
            ]),
            
            html.P("How it works:", className="mb-1 fw-bold"),
            html.P("The connections show the rating relationships between users and movies. The final recommendations are influenced by the strength of these connections."),
            
            html.H6("Hybrid Model", className="mt-3"),
            html.P("Combines both user-based and item-based approaches for better recommendations."),
            
            html.P("Key features:", className="mb-1 fw-bold"),
            html.Ul([
                html.Li("Adjust the weighting between models in the \"Additional Settings\" panel"),
                html.Li("See how different weightings affect your recommendations"),
                html.Li("Compare graph recommendations with the recommendation panel"),
            ]),
            
            html.H6("Neural Matrix Factorization (NeuMF)", className="mt-3"),
            html.P("A more advanced model that uses deep learning techniques."),
            
            html.P("What you see:", className="mb-1 fw-bold"),
            html.Ul([
                html.Li("User and item embeddings (mathematical representations)"),
                html.Li("Network layers showing how data flows through the model"),
                html.Li("Element-wise matrix multiplication combining different factors"),
                html.Li("Final prediction score"),
            ]),
            
            html.P("How to use:", className="mb-1 fw-bold"),
            html.P("Select different users and movies from the dropdown menus to explore how the model generates specific recommendations."),
            
            html.H5("Why This Visualization Helps", className="mt-3"),
            html.P("These interactive graphs make recommendation algorithms transparent by showing exactly why certain movies are suggested to you—whether it's because similar users enjoyed them or because the mathematical patterns in the neural network connected your profile to specific features."),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="modal",
    is_open=False,
    size="lg"
)


user_options = [{"label": f"User {user_id}", "value": user_id} for user_id in all_users]

neumf_user_options = [{"label": "User 52", "value": 52},
                      {"label": "User 132", "value": 132}, 
                      {"label": "User 307", "value": 307}, 
                      {"label": "User 358", "value": 358}]

model_options = [{"label": "User-based", "value": "ubcf"}, 
                 {"label": "Item-based", "value": "ibcf"},
                 {"label": "Neural", "value": "neumf"}]

node_count_options = [{"label": str(i), "value": i} for i in range(10, 101, 10)]

cytoscape_graph = [
    dbc.CardHeader(
        dbc.Row(
            [
                dbc.Col(html.H4("User-based Collaborative Filtering", id="graph-header", className="mb-0")),
                dbc.Col(dbc.Button("Reset Zoom", id="reset-view", size="sm", color="secondary"), width="auto", className="ms-auto"),
                dbc.Col(dbc.Button("Additional Settings", id="settings-button", size="sm", color="secondary"), width="auto", className="ms-auto"),
            ]
        )
    ),
    dbc.CardBody(
        [
            cyto.Cytoscape(
                id="cytoscape",
                elements=[],
                stylesheet=[],
                layout={"name": "preset"},
                style={
                    "width": "100%", 
                    "height": "600px",
                },
                minZoom=0.05,
                maxZoom=4,
            )
        ]
    ),
    dbc.CardFooter(
        dbc.Row(
            [
                dbc.Col(
                    html.Label("Model:", style={"fontWeight": "bold"}),
                    width="auto",
                    style={"paddingRight": "0rem", "marginRight": "0rem"}
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=model_options, 
                        value="ubcf", 
                        id="model-dropdown", 
                        clearable=False,
                        searchable=False,
                        className="dropdown-up dropdown-theme",
                        style={"minWidth": "175px", "borderWidth": "2px"}
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Label("Primary User:", style={"fontWeight": "bold"}),
                    width="auto",
                    style={"paddingRight": "0rem", "marginRight": "0rem"}
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=user_options, 
                        value=358, 
                        id="user-dropdown", 
                        clearable=False, 
                        searchable=False,
                        className="dropdown-up dropdown-theme",
                        style={"minWidth": "175px", "borderWidth": "2px"}
                    ), 
                    width="auto"
                ),
                dbc.Col(
                    html.Label("Movie:", style={"fontWeight": "bold"}),
                    width="auto",
                    style={"paddingRight": "0rem", "marginRight": "0rem"}
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="neumf-movie-dropdown",
                        options=[],
                        placeholder="N/A",
                        value=None,
                        clearable=False,
                        searchable=False,
                        optionHeight=50,
                        className="dropdown-up dropdown-theme",
                        style={"minWidth": "225px", "borderWidth": "2px"}
                    ),
                    width="auto",
                ),
            ], 
            align="center",
        ),
    ),
]

tooltip_container = html.Div(
    "TOOLTIP",
    id="graph-tooltip",
    style={
        'position': 'absolute',
        'backgroundColor': 'rgba(0, 0, 0, 0.8)',
        'color': 'white',
        'padding': '5px',
        'borderRadius': '8px',
        'visibility': 'hidden',
        'zIndex': 1000,
        }
)

corated_card = [
    dbc.CardHeader(html.H6(f"Click on a neighbor to see co-rated movies", id="corated-header")),
    dbc.CardBody(
        html.Div(
            dag.AgGrid(
                id="corated-grid",
                columnDefs=[
                    {
                        "headerName": "Movie", 
                        "field": "poster", 
                        "cellRenderer": "PosterWithTooltip",
                        "width": 100,
                    },
                    {
                        "headerName": "Primary Rating", 
                        "field": "primary_rating",
                        "cellRenderer": "StarRating",
                        "cellStyle": {
                            "display": "flex",
                            "alignItems": "center",
                        },
                        "flex": 1,
                    },
                    {
                        "headerName": "Neighbor Rating", 
                        "field": "neighbor_rating",
                        "cellRenderer": "StarRating",
                        "cellStyle": {
                            "display": "flex",
                            "alignItems": "center",
                        },
                        "flex": 1,
                    },
                ],
                rowData=[], # filled in callback
                className = "ag-theme-alpine",
                style={"height": "100%", "width": "100%"},
            ),
            style={
                "height": "350px",
                "overflowY": "auto", 
            }
        ),
        style={"padding": "3px"}
    )
]

ibcf_card = [
    dbc.CardHeader(html.H6(f"See help button for more information on the model.", id="ibcf-card-header")),
    dbc.CardBody()
]

neumf_card = [
    dbc.CardHeader(html.H6(f"Model Explanation", id="neumf-card-header")),
    dbc.CardBody([
        html.P([html.B("Introduction:") , " This visualization demonstrates how a model recommends specific movies to users by combining two complementary neural approaches. The left side shows 32-dimensional user and item embeddings, where each dot represents a latent feature value in both the MLP and GMF components."]),
        html.P([html.B("To the left of the embeddings lies the architecture of the MLP and GMF. Let’s walk through how each of these models work.")]),
        html.P([html.B("GMF (Pink Components):"), " Matrix factorization decomposes the sparse user-item interaction matrix into dense latent feature matrices. GMF enhances traditional MF by applying element-wise multiplication to user-item embeddings, then learning variable weights for each dimension and applying a non-linear activation function. The final 32-dimensional output is reflected on the right."]),
        html.P([html.B("MLP (Purple Components):"), " The multi-layer perceptron learns complex, non-linear user-item relationships by concatenating their embeddings and processing them through shrinking neural layers (128→64→32→16). Each layer with ReLU activation captures progressively more abstract interaction patterns that are impossible for linear models to learn. The final 16-dimensional output is reflected on the right."]),
        html.P([html.B("NeuMF:"), " The final model concatenates the GMF and MLP outputs, applying learned affine weights and bias before passing through a sigmoid function to produce a raw score between 0-1. This raw score is denormalized by multiplying by the maximum score in the dataset thus yielding the final predicted score for the film on the right."]),
        html.Hr(),  # Adds a horizontal line to separate sections
        html.P([html.B("Interactivity:")]),
        html.Ul([
            html.Li("Dot opacity represents the relative magnitude of values within each vector."),
            html.Li("Weight-to-sigmoid edges are blue for positive weights, red for negative weights."),
        ])
    ])
]



side_card = dbc.Card(
    id="side-card",
    children = corated_card,
    color="primary", 
    outline=True,
)

settings_offcanvas = html.Div(
    dbc.Offcanvas(
        [
            html.Hr(),
            html.H6("Graph Size:"),
            dcc.RadioItems(
                options = [
                    {"label":"Small", "value": 30},
                    {"label": "Medium", "value": 60},
                    {"label": "Large", "value": 100}
                ],
                value= 30,
                id="node-count-radio",
                inline=True,
                labelStyle={"marginRight": "20px"}
            ),
            html.Hr(),
            #html.Br(),
            html.H6("Number of recommendations:"),
            dcc.RadioItems(
                options = [
                    {"label":"5", "value": 5},
                    {"label": "10", "value": 10},
                    {"label": "15", "value": 15}
                ],
                value= 5,
                id="rec-count-radio",
                inline=True,
                labelStyle={"marginRight": "20px"}
            ),
            html.Hr(),
            html.H6("Model Weight:"),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "padding": "0px 5px 5px 5px"},
                children=[
                    html.Span("0% UBCF"),
                    html.Span("100% UBCF")
                ]
            ),
            dcc.Slider(
                id="model-weight-slider",
                min=0,
                max=1,
                step=0.1,
                value=0.5,
                marks={},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Hr(),
            
        ],
        id="settings-offcanvas",
        title="Additional Settings (UBCF and IBCF)",
        is_open=False,
    )
)

# ** LAYOUT **
app.layout = html.Div([
    dcc.Store(id="theme-store", data="light"),
    navbar,
    tooltip_container,
    modal,
    settings_offcanvas,
    html.Br(),
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(id="rec-panel", color="primary", outline=True, style={"maxHeight": "300px"}),
                            html.Br(),
                            side_card,
                            html.Br(),
                        ],
                        width=4
                    ),
                    dbc.Col(
                        dbc.Card(cytoscape_graph),
                        width=8
                    ),
                ]
            )
        ],
        style={"padding": "1.0"},
        fluid=True
    ),
    html.Div(id="dummy", style={"display": "none"}),
    html.Div(id="dummy2", style={"display": "none"}),
    html.Div(id="dummy3", style={"display": "none"}),
    html.Div(id="dummy4", style={"display": "none"}),
    dcc.Store(id="animation-trigger"),
    dcc.Store(id="id-iterator", data=0),
])


# ** APP CALLBACKS **
# Generate and set recommendations panel (top left)
@callback(
    Output("rec-panel", "children"),
    Input("user-dropdown", "value"),
    Input("node-count-radio", "value"),
    Input("model-dropdown", "value"),
    Input("model-weight-slider", "value"),
)
def set_rec_panel(prime_user, node_count, model, alpha_weight):
    if alpha_weight == 0:
        rec_model = "ibcf"
    elif alpha_weight == 100:
        rec_model = "ubcf"
    else:
        rec_model = "hybrid"  
        
    recommendations = cf.generate_recommendations(
        prime_user, 
        limit=5, 
        method=rec_model, 
        top_n=node_count,
        alpha=alpha_weight)
    return create_recommendation_panel(prime_user, recommendations, model, alpha_weight)



# Settings button to open offcanvas settings
@callback(
    Output("settings-offcanvas", "is_open"),
    Input("settings-button", "n_clicks"),
    prevent_initial_call=True,
)
def open_offcanvas(n_clicks):
    return True



# Set side card based on model selection
@callback(
    Output("side-card", "children"),
    Input("model-dropdown", "value"),
    Input("user-dropdown", "value"),
)
def set_side_card(model, user_id):
    if model == "ubcf":
        return corated_card
    elif model == "neumf":
        return neumf_card
    elif model == "ibcf":
        return ibcf_card



# Set NeuMF movie dropdown options
@callback(
    Output("neumf-movie-dropdown", "options"),
    Output("neumf-movie-dropdown", "value"),
    Input("user-dropdown", "value"),
    State("model-dropdown", "value"),
)
def set_neumf_movie_options(user_id, model):
    if model != 'neumf':
        return [], None
    options = get_movie_options(user_id)
    value = options[0]["value"] if options else None
    return options, value



# Change User-options based on model selection
@callback(
    Output("user-dropdown", "options"),
    Output("user-dropdown", "value"),
    Input("model-dropdown", "value"),
)
def set_user_options(model):
    if model == "neumf":
        return neumf_user_options, 358
    else:
        return user_options, 358



# Set graph header based on model selection
@callback(
    Output("graph-header", "children"),
    Input("model-dropdown", "value"),
)
def set_graph_header(model):
    if model == "ubcf":
        return "User-based Collaborative Filtering"
    elif model == "ibcf":
        return "Item-based Collaborative Filtering"
    elif model == "neumf":
        return "Neural Collaborative Filtering"
    else:
        return "Unknown Model"



# Generate cytoscape graph elements depending on model selection
@callback(
    Output("cytoscape", "elements"),
    Output("cytoscape", "stylesheet"),
    Output("animation-trigger", "data"),
    Output("id-iterator", "data"),
    Input("user-dropdown", "value"),
    Input("model-dropdown", "value"),
    State("id-iterator", "data"),
    Input("neumf-movie-dropdown", "value"),
    Input("node-count-radio", "value"),
    Input("rec-count-radio", "value"),
)
def set_graph_elements(prime_user, model, id_iteration, movie_id, node_count, rec_count):
    
    id_iteration += 1
    
    if model == "ibcf":
        elements = create_ibcf_elements(
            prime_user, 
            node_count, 
            node_count, 
            node_count, 
            rec_count, 
            cf, 
            id_iteration)
        ibcf_stylesheet = get_ibcf_stylesheet()
        return elements, ibcf_stylesheet, "True", id_iteration
    
    elif model == "neumf":
        elements = create_neumf_elements(prime_user, movie_id, id_iteration)
        neumf_stylesheet = get_neumf_stylesheet(id_iteration)
        return elements, neumf_stylesheet, "True", id_iteration
    
    elif model == "ubcf":
        ubcf_stylesheet = get_ubcf_stylesheet()
        elements = create_ubcf_elements(
            prime_user=prime_user, 
            profile_limit=node_count, 
            neighbor_limit=node_count, 
            rec_limit=rec_count, 
            rec_top_n=node_count, 
            cf=cf, 
            id_iteration=id_iteration)
        return elements, ubcf_stylesheet, "True", id_iteration
    
    else:
        return [], [], "False", id_iteration



# UBCF animations and graph interactions
clientside_callback(
    """
    function(elements, model, node_count) {
        
        if (model === "neumf") {
            return null;
        }
        
        console.log("Callback triggered - UBCF");
        // Get the Cytoscape instance from the DOM
        var cytoscapeDiv = document.getElementById('cytoscape');
        var cy = cytoscapeDiv && cytoscapeDiv._cyreg ? cytoscapeDiv._cyreg.cy : null;
        
        if (cy) {
            cy.ready(function() {
                
                // Initialize view
                setTimeout(function() {
                    
                    if (node_count === 30) {
                        cy.zoom(0.30);
                        cy.pan({ x: 30, y: 18 });
                    } else if (node_count === 60) {
                        cy.zoom(0.15);
                        cy.pan({ x: 111, y: 16 });
                    } else if (node_count === 100) {
                        cy.zoom(0.08);
                        cy.pan({ x: 183, y: 36 });
                    }
                    
                }, 100);
                
                
                
                // ANIMATION 1 - Make nodes visible, slide in profile nodes
                setTimeout(function() {
                    cy.nodes('.prime_user').style('visibility', 'visible');
                    cy.nodes('.profile').style('visibility', 'visible');
                    cy.nodes('.neighbor').style('visibility', 'visible');
                    cy.nodes('.rec').style('visibility', 'visible');
                    
                    // Animate prime_user if not already animated
                    cy.nodes('.prime_user').filter(function(node) {
                        return !node.hasClass('animated');
                    }).forEach(function(node) {
                        var finalPos = node.data().final_position;
                        node.animate({
                            position: { x: finalPos.x, y: finalPos.y },
                            duration: 500
                        });
                        node.addClass('animated');
                    });
                    
                    // Animate profile nodes if not already animated
                    cy.nodes('.profile').filter(function(node) {
                        return !node.hasClass('animated');
                    }).forEach(function(node) {
                        var finalPos = node.data().final_position;
                        node.animate({
                            position: { x: finalPos.x, y: finalPos.y },
                            duration: 2000
                        });
                        node.addClass('animated');
                    });
                }, 100);
                
                
                
                // ANIMATION 2 - Slide in neighbor nodes
                setTimeout(function() {
                    cy.nodes('.neighbor').filter(function(node) {
                        return !node.hasClass('animated');
                    }).forEach(function(node) {
                        var finalPos = node.data().final_position;
                        node.animate({
                            position: { x: finalPos.x, y: finalPos.y },
                            duration: 2000
                        });
                        node.addClass('animated');
                    });
                }, 1000);
                
                
                
                // ANIMATION 3 - Slide in recommendation nodes
                setTimeout(function() {
                    cy.nodes('.rec').filter(function(node) {
                        return !node.hasClass('animated');
                    }).forEach(function(node) {
                        var finalPos = node.data().final_position;
                        node.animate({
                            position: { x: finalPos.x, y: finalPos.y },
                            duration: 2000
                        });
                        node.addClass('animated');
                    });
                }, 1500);
                
                
                
                // ANIMATION 4 - Animate edges one layer at a time
                function animateEdges(edgeSelector, delay) {
                    setTimeout(function() {
                        cy.edges(edgeSelector).filter(function(edge) {
                            return !edge.hasClass('animated');
                        }).forEach(function(edge) {
                            var rating = edge.data('rating');
                            var color = rating <= 1.5 ? '#ff003e' :
                                        rating <= 2.5 ? '#ff0060' :
                                        rating <= 3.5 ? '#d933c2' :
                                        rating <= 4.0 ? '#b84edd' : 
                                        rating <= 4.5 ? '#8a63f1' :
                                        rating <= 5.0 ? '#4073ff' : '#4073ff';
                            edge.animate({
                                style: {
                                    'opacity': 1,
                                    'line-color': color
                                },
                                duration: 1000
                            });
                            edge.addClass('animated');
                        });
                    }, delay);
                }
                
                animateEdges('.profile_edge', 4000);
                animateEdges('.crowd_profile_edge', 5000);
                animateEdges('.crowd_rec_edge', 6000);
                
                // Fade out edges after animation
                setTimeout(function() {
                    cy.edges('.cf_edge').filter(function(edge) {
                        return !edge.hasClass('finalanimated');
                    }).forEach(function(edge) {
                        var rating = edge.data('rating');
                        var color = rating <= 1.5 ? '#ff003e' :
                                    rating <= 2.5 ? '#ff0060' :
                                    rating <= 3.5 ? '#d933c2' :
                                    rating <= 4.0 ? '#b84edd' : 
                                    rating <= 4.5 ? '#8a63f1' :
                                    rating <= 5.0 ? '#4073ff' : '#4073ff';
                        edge.animate({
                            style: { 
                                'opacity': 0.05,
                                'line-color': color
                            },
                            duration: 500
                        });
                        edge.addClass('finalanimated');
                    });
                }, 8000);
                
                
                
                // HOVER INTERACTIONS - Show tooltip on node/edge hover
                var tooltip = document.getElementById('graph-tooltip');
                // clear existing listeners
                cy.off('mouseout');
                cy.off('mouseover');
                
                // Display tooltip on hover
                cy.on('mouseover', 'node, edge', function(event) {
                    var ele = event.target;
                    var data = ele.data();
                    
                    let content = '';
                    if (ele.isNode && ele.isNode()) {
                        
                        if (model === "ibcf") {
                            content = `${data.label || data.id}`;
                        } else if (ele.hasClass('neighbor')) {
                            let similarity = Number(data.similarity).toFixed(3);
                            content = `Neighbor: ${data.label || data.id}\nSimilarity to primary = ${similarity}\nNum. co-rated movies = ${data.num_corated}`;
                        } else {
                            content = `${data.label || data.id}`;
                        }
                    } else if (ele.isEdge && ele.isEdge() && ele.hasClass('highlighted')) {
                        
                        if (model === "ibcf") {
                            content = `Edge: ${data.source} → ${data.target}`;
                        } else if (ele.hasClass('profile_edge')) {
                            let targetLabel = cy.getElementById(data.target).data('label') || '';
                            let shortLabel = targetLabel.length > 25 ? targetLabel.substring(0, 25) + '…' : targetLabel;
                            content = `${shortLabel}\nPrimary User Rating = ${data.rating}`;
                        } else if (ele.hasClass('crowd_profile_edge')) {
                            let targetLabel = cy.getElementById(data.target).data('label') || '';
                            let shortLabel = targetLabel.length > 25 ? targetLabel.substring(0, 25) + '…' : targetLabel;
                            let sourceLabel = cy.getElementById(data.source).data('label') || '';
                            content = `${shortLabel}\n${sourceLabel} Rating = ${data.rating}`;
                        } else if (ele.hasClass('crowd_rec_edge')) {
                            let targetLabel = cy.getElementById(data.target).data('label') || '';
                            let shortLable = targetLabel.length > 25 ? targetLabel.substring(0, 25) + '…' : targetLabel;
                            let sourceLabel = cy.getElementById(data.source).data('label') || '';
                            let rating = data.rating;
                            let similarity = Number(data.similarity).toFixed(3);
                            content = `${shortLable}\n${sourceLabel} Rating = ${rating}\n${sourceLabel} similarity to primary = ${similarity}`;
                        } else {
                            content = `Edge: ${data.source} → ${data.target}`;
                        }
                        
                        
                    }
                    
                    if (content) {
                        tooltip.textContent = content;
                        tooltip.style.visibility = 'visible';
                    }
                });
                    
                cy.on('mouseout', 'node, edge', function(event) {
                    tooltip.style.visibility = 'hidden';
                });
                
                cy.on('mousemove', function(event) {
                    const evt = event.originalEvent;
                    const tooltipWidth = tooltip.offsetWidth;
                    const pageWidth = window.innerWidth;
                    let left = evt.pageX + 10;
                    if (left + tooltipWidth > pageWidth) {
                        left = evt.pageX - tooltipWidth - 10; // Display to the left if too close to the right
                    }
                    tooltip.style.left = `${left}px`;
                    tooltip.style.top = `${evt.pageY + 10}px`;
                });
                
                
                
                // CLICK INTERACTIONS - Select nodes to highlight edges
                cy.on('tap', 'node', function(event) {
                var clickedNode = event.target;
                
                // Skip if animations are running
                if (cy.$(':animated').length > 0) return;
                
                // If this node is already selected, deselect it
                if (clickedNode.hasClass('selected')) {
                    clickedNode.removeClass('selected');
                    
                    // Reset only the edges connected to this node
                    clickedNode.connectedEdges().forEach(function(edge) {
                        var rating = edge.data('rating');
                        var color = rating <= 1.5 ? '#ff003e' :
                                    rating <= 2.5 ? '#ff0060' :
                                    rating <= 3.5 ? '#d933c2' :
                                    rating <= 4.0 ? '#b84edd' : 
                                    rating <= 4.5 ? '#8a63f1' :
                                    rating <= 5.0 ? '#4073ff' : '#4073ff';
                        edge.animate(
                            { style: { 'opacity': 0.05, 'line-color': color } },
                            { duration: 200 }
                        );
                        edge.removeClass('highlighted');
                        
                        // Reset the edges connected to prime user
                        var otherNode = edge.source().id() === clickedNode.id() ? edge.target() : edge.source();
                        if (otherNode.hasClass('profile')) {
                            otherNode.connectedEdges().forEach(function(e) {
                                var targetNode = e.source().id() === otherNode.id() ? e.target() : e.source();
                                if (targetNode.hasClass('prime_user')) {
                                    var rating = e.data('rating');
                                    var color = rating <= 1.5 ? '#ff003e' :
                                                rating <= 2.5 ? '#ff0060' :
                                                rating <= 3.5 ? '#d933c2' :
                                                rating <= 4.0 ? '#b84edd' : 
                                                rating <= 4.5 ? '#8a63f1' :
                                                rating <= 5.0 ? '#4073ff' : '#4073ff';
                                    e.animate(
                                        { style: { 'opacity': 0.05, 'line-color': color } },
                                        { duration: 200 }
                                    );
                                    e.removeClass('highlighted');
                                }
                            });
                        }
                    });
                    return;
                }
                
                // Otherwise, select this node and highlight connected edges
                cy.nodes().removeClass('selected'); // Remove selection from other nodes
                clickedNode.addClass('selected'); // Add selection to the clicked node
                
                // Reset all edges to low opacity
                cy.edges().forEach(function(edge) {
                    if (!edge.hasClass('highlighted')) {
                        var rating = edge.data('rating');
                        var color = rating <= 1.5 ? '#ff003e' :
                                    rating <= 2.5 ? '#ff0060' :
                                    rating <= 3.5 ? '#d933c2' :
                                    rating <= 4.0 ? '#b84edd' : 
                                    rating <= 4.5 ? '#8a63f1' :
                                    rating <= 5.0 ? '#4073ff' : '#4073ff';
                        edge.animate(
                            { style: { 'opacity': 0.05, 'line-color': color } },
                            { duration: 200 }
                        );
                        edge.removeClass('highlighted');
                    }
                });
                
                // Highlight directly connected edges
                var connectedEdges = clickedNode.connectedEdges();
                connectedEdges.forEach(function(edge) {
                    var source = edge.source();
                    var target = edge.target();
                    
                    edge.animate({style: { 'opacity': 1 }}, { duration: 200 });
                    edge.addClass('highlighted');
                    
                    // Scan for 'profile' nodes on the other side
                    if (clickedNode.hasClass('neighbor')) {
                        [source, target].forEach(function(node) {
                            if (node.hasClass('profile')) {
                                // Now find edges from this profile node to the primary node
                                var extraEdges = node.connectedEdges().filter(function(e) {
                                    var otherNode = e.source().id() === node.id() ? e.target() : e.source();
                                    return otherNode.hasClass('prime_user');
                                });
                                extraEdges.forEach(function(e) {
                                    var rating = e.data('rating');
                                    var color = rating <= 1.5 ? '#ff003e' :
                                                rating <= 2.5 ? '#ff0060' :
                                                rating <= 3.5 ? '#d933c2' :
                                                rating <= 4.0 ? '#b84edd' : 
                                                rating <= 4.5 ? '#8a63f1' :
                                                rating <= 5.0 ? '#4073ff' : '#4073ff';
                                    e.animate(
                                        { style: { 'opacity': 1, 'line-color': color } },
                                        { duration: 200 }
                                    );
                                    e.addClass('highlighted');
                                });
                            }
                        });
                    }
                });
                
                // Reset on background click
                cy.on('tap', function(event) {
                    if (event.target === cy && cy.$(':animated').length === 0) {
                        cy.edges().forEach(function(edge) {
                            if (edge.hasClass('highlighted')) {
                                edge.animate({
                                    style: {
                                        'opacity': 0.05
                                    },
                                    duration: 200
                                });
                                edge.removeClass('highlighted');
                            }
                        });
                    }
                });
            });
                
            }); 
        } else {
            console.error("Cytoscape instance not found under _cyreg.cy");
        }
        return null;
    }
    """,
    Output('dummy', 'children'),
    Input('animation-trigger', 'data'),
    State("model-dropdown", "value"),
    State("node-count-radio", "value"),
);



# NeuMF animations and graph interactions
clientside_callback(
    """
    function(elements, model) {
        
        
        if (model !== "neumf") {
            return null;
        }
        
        console.log("Callback triggered - NeuMF");
        
        var cytoscapeDiv = document.getElementById('cytoscape');
        var cy = cytoscapeDiv && cytoscapeDiv._cyreg ? cytoscapeDiv._cyreg.cy : null;
        
        if (cy) {
            cy.ready(function() {
                var tooltip = document.getElementById('graph-tooltip');
                cy.off('mouseout');
                cy.off('mouseover');
                cy.off('tap')
                cy.off('click')
                
                
                cy.on('mouseover', 'node, edge', function(event) {
                    var ele = event.target;
                    var data = ele.data();
                    
                    let content = '';
                    if (ele.isNode && ele.isNode()) {
                        
                        if (ele.hasClass('mini_circle') && !ele.hasClass('mlp_layer_mini_circle')) {
                            let value = data.value;
                            let formattedValue;
                            if (value !== undefined) {
                                // Check if the value is 0 or close to 0
                                if (value === 0) {
                                    formattedValue = '0';  // Special case for zero
                                }
                                // Check if the number is small enough to need scientific notation
                                else if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e6) {
                                    // Format with 3 significant digits in scientific notation
                                    formattedValue = value.toExponential(2);  // 3 significant digits
                                } else {
                                    // Format as a regular number with 3 significant digits
                                    formattedValue = value.toPrecision(3);  // 3 significant digits, no scientific notation unless needed
                                }
                            } else {
                                formattedValue = '(no value)';
                            }
                            
                            content = `Value: ${formattedValue}`;
                        }
                        if (ele.hasClass('top_node')) {
                            let label = data.label;
                            
                            content = `${label}`;
                        }
                        if (ele.hasClass('neumf_yhat')) {
                            content = `Predicted Rating`;
                        }
                        if (ele.hasClass('embed_mini_circle')) {
                            let mlp_value = data.value;
                            let mf_value = data.mf_value;
                            
                            let mlp_formattedValue;
                            if (mlp_value !== undefined) {
                                // Check if the value is 0 or close to 0
                                if (mlp_value === 0) {
                                    mlp_formattedValue = '0';  // Special case for zero
                                }
                                // Check if the number is small enough to need scientific notation
                                else if (Math.abs(mlp_value) < 1e-3 || Math.abs(mlp_value) > 1e6) {
                                    // Format with 3 significant digits in scientific notation
                                    mlp_formattedValue = mlp_value.toExponential(2);  // 3 significant digits
                                } else {
                                    // Format as a regular number with 3 significant digits
                                    mlp_formattedValue = mlp_value.toPrecision(3);  // 3 significant digits, no scientific notation unless needed
                                }
                            } else {
                                mlp_formattedValue = '(no value)';
                            }
                            let mf_formattedValue;
                            if (mf_value !== undefined) {
                                // Check if the value is 0 or close to 0
                                if (mf_value === 0) {
                                    mf_formattedValue = '0';  // Special case for zero
                                }
                                // Check if the number is small enough to need scientific notation
                                else if (Math.abs(mf_value) < 1e-3 || Math.abs(mf_value) > 1e6) {
                                    // Format with 3 significant digits in scientific notation
                                    mf_formattedValue = mf_value.toExponential(2);  // 3 significant digits
                                } else {
                                    // Format as a regular number with 3 significant digits
                                    mf_formattedValue = mf_value.toPrecision(3);  // 3 significant digits, no scientific notation unless needed
                                }
                            } else {
                                mf_formattedValue = '(no value)';
                            }
                            
                            content = `MLP Embedding Value: ${mlp_formattedValue}\nMF Embedding Value: ${mf_formattedValue}`;
                        }
                        
                    } else if (ele.isEdge && ele.isEdge() && ele.hasClass('highlighted')) {
                        content = `Edge: ${data.source} → ${data.target}`;
                    } 

                    if (content) {
                        tooltip.textContent = content;
                        tooltip.style.visibility = 'visible';
                    }
                });

                cy.on('mouseout', 'node, edge', function(event) {
                    tooltip.style.visibility = 'hidden';
                });

                cy.on('mousemove', function(event) {
                    const evt = event.originalEvent;
                    const tooltipWidth = tooltip.offsetWidth;
                    const pageWidth = window.innerWidth;
                    let left = evt.pageX + 10;
                    if (left + tooltipWidth > pageWidth) {
                        left = evt.pageX - tooltipWidth - 10;
                    }
                    tooltip.style.left = `${left}px`;
                    tooltip.style.top = `${evt.pageY + 10}px`;
                });
            });
        }

        return null;
    }
    """,
    Output("dummy3", "children"),
    Input("animation-trigger", "data"),
    State("model-dropdown", "value")
)



# Update AG Grid theme based on dark mode
@callback(
    Output("corated-grid", "className"),
    Input("theme-store", "data"),
)
def update_grid_theme(current_theme):
    return "ag-theme-alpine-dark" if current_theme == "dark" else "ag-theme-alpine"



# Display co-rated movies in AG Grid
@callback(
    Output("corated-grid", "rowData"),
    Output("corated-header", "children"),
    Input("cytoscape", "tapNode"),
    Input("user-dropdown", "value"),
    Input("model-dropdown", "value"),
)
def display_node_click(nodeData, prime_user, model):
    if model != "ubcf":
        return [], "Change to User-based CF to see co-rated movies"
    
    
    if nodeData is None or "neighbor" not in nodeData["classes"]:
        return [], "Click on a neighbor to see co-rated movies"
    
    neighbor_id = int(nodeData["data"]["id"].split("_")[1])
    corated_movies = cf.get_corated(prime_user, neighbor_id)
    if corated_movies.empty:
        return "No common movies found"
    
    # Build rowData for AG Grid
    rows = []
    for _, row in corated_movies.iterrows():
        poster_url = get_movie_poster(row["movieId"])
        rows.append({
            "poster": {
                "src": poster_url,
                "title": row["title"]
            },
            "primary_rating": f'{row[f"rating_{prime_user}"]:.1f}',
            "neighbor_rating": f'{row[f"rating_{neighbor_id}"]:.1f}',
        })
    
    new_header = f"Co-rated movies with User {neighbor_id}"
    
    return rows, new_header



# Dark mode switch
clientside_callback(
    """
    function(dark) {
        document.documentElement.setAttribute("data-bs-theme", dark ? "dark" : "light");
        return dark ? "dark" : "light";
    }
    """,
    Output("theme-store", "data"),
    Input("theme-switch", "value")
)



# Reset view button
clientside_callback(
    """
    function(n_clicks, model, node_count) {
        
        var cytoscapeDiv = document.getElementById('cytoscape');
        var cy = cytoscapeDiv && cytoscapeDiv._cyreg ? cytoscapeDiv._cyreg.cy : null;
        
        
        
        if (cy) {
            if (model !== "neumf") {
                
                if (node_count === 30) {
                    cy.animate({
                        pan: { x: 30, y: 18 },
                        zoom: 0.30
                    }, {
                        duration: 200
                    });
                } else if (node_count === 60) {
                    cy.animate({
                        pan: { x: 111, y: 16 },
                        zoom: 0.15
                    }, {
                        duration: 200
                    });
                } else if (node_count === 100) {
                    cy.animate({
                        pan: { x: 183, y: 36 },
                        zoom: 0.08
                    }, {
                        duration: 200
                    });
                }
            }
            if (model === "neumf") {
                cy.animate({
                    pan: { x: 175, y: 80 },
                    zoom: 0.54
                }, {
                    duration: 200
                });
            }
        }
    }
    """,
    Output("dummy2", "children"),
    Input("reset-view", "n_clicks"),
    State("model-dropdown", "value"),
    State("node-count-radio", "value"),
    prevent_initial_call=True
)



# Modal toggles
@callback(
    Output("modal", "is_open"),
    Input("about-button", "n_clicks"),
    Input("close-modal", "n_clicks"),
    State("modal", "is_open"),
)
def toggle_modal(open_click, close_click, is_open):
    if open_click > 0 and not is_open:
        return True
    elif close_click > 0 and is_open:
        return False
    return is_open
    
# END OF CALLBACKS

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, port=8050)