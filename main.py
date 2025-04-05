import osmnx as ox
from osmnx import settings
import pandas as pd
from shapely.geometry import shape
import math
import os
import tempfile
import networkx as nx
import folium
from folium.plugins import MarkerCluster
import logging
from sklearn.cluster import KMeans
import numpy as np
import random
import multiprocessing
import streamlit as st
import streamlit.components.v1 as components
from deap import base, creator, tools, algorithms
from streamlit_folium import st_folium
from shapely.geometry import box
from opencage.geocoder import OpenCageGeocode
import time
from geopy.distance import geodesic
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image

# Enable logging
logging.basicConfig(level=logging.INFO)

settings.max_query_area_size = 2500000000


def fetch_osm_data(bbox_tuple):
    try:
        logging.info(f"Fetching OSM data for bbox: {bbox_tuple}")

        # Define tags to fetch everything in a single request
        tags = {
            "highway": True,  # Roads
            "landuse": True,  # Land use
            "natural": "water",  # Water bodies
            "place": True  # Places
        }

        # Fetch features (roads, land use, water, places)
        features = ox.features_from_bbox(bbox_tuple, tags=tags)  # Fetch all in one request

        # Fetch road network (simplify=False for speed)
        roads = ox.graph_from_bbox(bbox_tuple, network_type='drive', simplify=False)

        # Separate features into specific categories
        roads_data = features.get('highway', [])
        land_use = features.get('landuse', [])
        water_bodies = features.get('natural', [])
        places = features.get('place', [])

        return roads, roads_data, land_use, water_bodies, places
    except Exception as e:
        logging.error(f"Error fetching OSM data: {e}")
        return None, None, None, None, None



def create_map_from_bbox(bbox):
    # Create a map centered around the middle of the bounding box
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Create the folium map object centered on the bounding box
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Set the map's bounds to the bounding box
    folium_map.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])

    return folium_map

def capture_map_as_image(bbox):
    # Create the map centered around the bounding box
    map_object = create_map_from_bbox(bbox)

    # Save the folium map to HTML in a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        map_path = tmp_file.name
        map_object.save(map_path)

    # Set up Selenium WebDriver to open the saved map HTML
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920x1080")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"file://{os.path.abspath(map_path)}")  # Use absolute path for the HTML file

    # Wait for the map to load
    time.sleep(5)
    driver.save_screenshot("map_screenshot.png")
    driver.quit()

    # Open the image and get its dimensions
    image = Image.open("map_screenshot.png")
    img_width, img_height = image.size  # Get the image dimensions

    # Return the image path
    return "map_screenshot.png"


# ===========================
# 2️⃣ Create Graph Representation
# ===========================
def create_graph_from_osm_data(roads, land_use, water_bodies, places):
    try:
        G = nx.Graph()

        # Ensure inputs are DataFrames
        if isinstance(roads, pd.Series):
            roads = roads.to_frame()
        if isinstance(land_use, pd.Series):
            land_use = land_use.to_frame()
        if isinstance(water_bodies, pd.Series):
            water_bodies = water_bodies.to_frame()
        if isinstance(places, pd.Series):
            places = places.to_frame()

        # Debugging: Print dataframe columns
        print("Roads columns:", roads.columns if isinstance(roads, pd.DataFrame) else "Not a DataFrame")
        print("Land Use columns:", land_use.columns if isinstance(land_use, pd.DataFrame) else "Not a DataFrame")
        print("Water Bodies columns:", water_bodies.columns if isinstance(water_bodies, pd.DataFrame) else "Not a DataFrame")
        print("Places columns:", places.columns if isinstance(places, pd.DataFrame) else "Not a DataFrame")

        # Add nodes and attributes from road network
        if isinstance(roads, pd.DataFrame) and "geometry" in roads.columns:
            for idx, row in roads.iterrows():
                if row.geometry and hasattr(row.geometry, "centroid"):
                    G.add_node(idx, x=row.geometry.centroid.x, y=row.geometry.centroid.y, type="road")

        # Add land use zones
        if isinstance(land_use, pd.DataFrame) and "geometry" in land_use.columns:
            for idx, row in land_use.iterrows():
                if row.geometry and hasattr(row.geometry, "centroid"):
                    G.add_node(idx, x=row.geometry.centroid.x, y=row.geometry.centroid.y, type="landuse")

        # Add water bodies
        if isinstance(water_bodies, pd.DataFrame) and "geometry" in water_bodies.columns:
            for idx, row in water_bodies.iterrows():
                if row.geometry and hasattr(row.geometry, "centroid"):
                    G.add_node(idx, x=row.geometry.centroid.x, y=row.geometry.centroid.y, type="water")

        # Add places
        if isinstance(places, pd.DataFrame) and "geometry" in places.columns:
            for idx, row in places.iterrows():
                if row.geometry and hasattr(row.geometry, "centroid"):
                    G.add_node(idx, x=row.geometry.centroid.x, y=row.geometry.centroid.y, type="place")

        return G

    except Exception as e:
        logging.error(f"Error creating graph: {e}")
        print(f"Error creating graph: {e}")  # Ensure it prints in the console
        return None




# ===========================
# 3️⃣ Scoring Function for Optimization
# ===========================

def haversine_distance(coord1, coord2):
    """Calculate distance between two lat/lon coordinates in meters."""
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

import math

def haversine_distance(coord1, coord2):
    """Calculate distance between two lat/lon coordinates in meters."""
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_score(index_zone_tuple, graph, constraints, land_use, water_bodies, places, roads_data, full_individual=None):
    """Enhanced scoring function for urban zoning optimization with clustering, buffers, dispersion, and walkability."""
    node_index, zone = index_zone_tuple
    score = 0

    # Convert index to actual node ID
    nodes_list = list(graph.nodes)
    if node_index >= len(nodes_list):
        return 0
    node_id = nodes_list[node_index]

    # Retrieve coordinates for dispersion logic
    coords = None
    node_data = graph.nodes[node_id]
    if 'x' in node_data and 'y' in node_data:
        coords = (node_data['y'], node_data['x'])

    # Road access
    if "road_access" in constraints:
        road_score = graph.degree(node_id)
        if zone == "commercial":
            score += constraints["road_access"] * road_score
        elif zone == "residential":
            score += (constraints["road_access"] * road_score) * 0.8

    # Land use compatibility
    if node_id in land_use:
        land_type = land_use[node_id]
        if zone == "residential" and land_type in ["park", "forest"]:
            score += constraints.get("green_space", 1.0) * 2
        if zone == "industrial":
            if land_type in ["park", "residential"]:
                score -= constraints.get("green_space", 1.0) * 3
            if land_type == "industrial":
                score += constraints.get("industrial_clustering", 1.0)

    # Industrial buffer
    if "industrial_buffer" in constraints:
        for neighbor in graph.neighbors(node_id):
            if neighbor in land_use and land_use[neighbor] == "industrial":
                if zone in ["residential", "commercial"]:
                    score -= constraints["industrial_buffer"]
                elif zone == "park":
                    score += constraints["industrial_buffer"]

    # Water body proximity
    if node_id in water_bodies:
        if zone == "industrial":
            score -= constraints.get("water_proximity", 1.0) * 3
        elif zone == "residential":
            score += constraints.get("water_proximity", 1.0) * 2

    # Urban placement
    if node_id in places:
        place_type = places[node_id]
        if zone == "commercial":
            if place_type in ["city", "town"]:
                score += constraints.get("urban_placement", 1.0) * 3
            elif place_type == "village":
                score -= constraints.get("urban_placement", 1.0) * 2
        elif zone == "residential":
            if place_type in ["village", "town"]:
                score += constraints.get("urban_placement", 1.0) * 2

    # Mixed-use encouragement
    if "mixed_use" in constraints:
        for neighbor in graph.neighbors(node_id):
            if neighbor in land_use:
                if zone == "commercial" and land_use[neighbor] == "residential":
                    score += constraints["mixed_use"]
                elif zone == "residential" and land_use[neighbor] == "commercial":
                    score += constraints["mixed_use"]

    # Walkability bonus
    if "walkability" in constraints:
        if node_id in roads_data:
            if zone in ["residential", "commercial"]:
                score += constraints["walkability"]

    # Amenities near residential zones
    if "amenities" in constraints and node_id in places:
        if zone == "residential" and places[node_id] in ["school", "hospital"]:
            score += constraints["amenities"]

    # Education near commercial
    if zone == "educational":
        for neighbor in graph.neighbors(node_id):
            if neighbor in land_use and land_use[neighbor] == "commercial":
                score += constraints.get("edu_near_commercial", 5)

    # 💡 Dispersion penalty for commercial & residential zones
    if full_individual and coords:
        same_zone_coords = []
        for i, z in enumerate(full_individual):
            if z == zone and i != node_index:
                other_id = nodes_list[i]
                other_data = graph.nodes[other_id]
                if 'x' in other_data and 'y' in other_data:
                    same_zone_coords.append((other_data['y'], other_data['x']))

        if same_zone_coords:
            distances = [haversine_distance(coords, other_coord) for other_coord in same_zone_coords]
            avg_distance = sum(distances) / len(distances)

            if zone in ["residential", "commercial"]:
                threshold = 500  # meters
                if avg_distance > threshold:
                    penalty = (avg_distance - threshold) / 100
                    score -= penalty * constraints.get("dispersion_penalty", 1.0)

    return score


def densify_graph(graph, segments=2, min_edge_length=50):
    new_graph = graph.copy()
    next_node_id = 1_000_000

    for u, v in list(graph.edges):
        if 'x' in graph.nodes[u] and 'y' in graph.nodes[u] and 'x' in graph.nodes[v] and 'y' in graph.nodes[v]:
            x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
            x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
            dist = haversine_distance(x1, y1, x2, y2)

            if dist < min_edge_length:
                continue  # Skip short edges

            prev = u
            for i in range(1, segments):
                t = i / segments
                mx, my = x1 * (1 - t) + x2 * t, y1 * (1 - t) + y2 * t
                synthetic_id = next_node_id
                new_graph.add_node(synthetic_id, x=mx, y=my)
                new_graph.add_edge(prev, synthetic_id)
                prev = synthetic_id
                next_node_id += 1

            new_graph.add_edge(prev, v)
            new_graph.remove_edge(u, v)

    return new_graph

# ===========================
# 4️⃣ Genetic Algorithm for Zoning Optimization
# ===========================

def evaluate(individual, graph, constraints, land_use, water_bodies, places, roads_data):
    total_score = 0
    penalty = 0

    zone_types = {
        0: "residential",
        1: "commercial",
        2: "industrial",
        3: "educational",
        4: "park"
    }

    node_list = list(graph.nodes)
    zone_map = {}  # {zone_type: [node IDs]}

    for idx, zone_idx in enumerate(individual):
        node_id = node_list[idx]
        zone = zone_types.get(zone_idx, "unknown")
        zone_map.setdefault(zone, []).append(node_id)

        # Base score
        score = calculate_score((idx, zone), graph, constraints, land_use, water_bodies, places, roads_data)
        total_score += score

        # Neighbor constraints
        neighbors = list(graph.neighbors(node_id))
        for neighbor in neighbors:
            if neighbor not in node_list:
                continue
            neighbor_idx = node_list.index(neighbor)
            neighbor_zone = zone_types.get(individual[neighbor_idx], "unknown")

            # Penalize industrial near sensitive zones
            if zone == "industrial" and neighbor_zone in ["residential", "commercial", "educational"]:
                penalty += 10

            # Encourage green buffers
            if zone == "industrial" and neighbor_zone != "park":
                penalty += 5

        # Water constraint
        if node_id in water_bodies and zone != "park":
            penalty += 15

    # Encourage distributed green space
    park_nodes = zone_map.get("park", [])
    if len(park_nodes) > 1:
        coords = np.array([[graph.nodes[n].get('x', 0), graph.nodes[n].get('y', 0)] for n in park_nodes])
        dist_var = np.var(coords, axis=0).mean()
        total_score += dist_var * 5

    # Penalize large homogeneous zones
    total_nodes = len(graph.nodes)
    for zone, nodes in zone_map.items():
        if len(nodes) > total_nodes * 0.3:
            penalty += (len(nodes) - total_nodes * 0.3) * 2

    # Penalize fragmentation
    for zone, nodes in zone_map.items():
        subgraph = graph.subgraph(nodes)
        num_components = nx.number_connected_components(subgraph)
        penalty += num_components

    # Optional: Encourage smoother transitions between zones
    transition_penalty_matrix = {
        ("residential", "industrial"): 10,
        ("educational", "industrial"): 8,
        ("residential", "commercial"): -2,  # Reward good transition
        ("park", "residential"): -1,
    }

    for idx, zone_idx in enumerate(individual):
        node_id = node_list[idx]
        zone = zone_types.get(zone_idx, "unknown")
        for neighbor in graph.neighbors(node_id):
            if neighbor not in node_list:
                continue
            neighbor_idx = node_list.index(neighbor)
            neighbor_zone = zone_types.get(individual[neighbor_idx], "unknown")
            penalty += transition_penalty_matrix.get((zone, neighbor_zone), 0)

    return (total_score - penalty,)


# ---------------- Main Optimization Function ----------------
def optimize_zoning(graph, num_zones, constraints, osm_data):
    """
    Enhanced Genetic Algorithm for optimal zoning.
    Includes elitism, adaptive mutation, and constraint handling.
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def initialize_clustered_zones(graph, num_zones):
        """
        Initializes zoning using K-Means clustering for better spatial grouping.
        """
        node_positions = []
        node_ids = []
        for node, data in graph.nodes(data=True):
            if 'x' in data and 'y' in data:
                node_positions.append((data['x'], data['y']))
            else:
                node_positions.append((random.uniform(0, 1), random.uniform(0, 1)))  # Assign random position
            node_ids.append(node)

        node_positions = np.array(node_positions)  # Convert to NumPy array after ensuring all values exist

        kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(node_positions)

        # Map clustering results to node order
        initial_zoning = [cluster_labels[node_ids.index(node)] if node in node_ids else random.choice(cluster_labels)
                          for node in graph.nodes]

        return initial_zoning

    def init_clustered_individual():
        return creator.Individual(initialize_clustered_zones(graph, num_zones))

    toolbox.register("individual", init_clustered_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Extract data from OSM
    land_use = osm_data.get("land_use", {})
    water_bodies = osm_data.get("water_bodies", {})
    places = osm_data.get("places", {})
    roads_data = osm_data.get("roads_data", {})

    # Register evaluate function with additional args
    toolbox.register("evaluate", evaluate, graph=graph, constraints=constraints,
                     land_use=land_use, water_bodies=water_bodies, places=places, roads_data=roads_data)

    # ---------------- Genetic Algorithm Operators ----------------
    toolbox.register("mate", tools.cxTwoPoint)

    def mutate_clustered(individual, graph, mutation_prob=0.1, num_zones=5):
        """
        Encourages smoother zoning transitions while preserving spatial logic.
        Randomly changes zone assignment based on neighborhood influence.
        """
        node_list = list(graph.nodes)

        for i in range(len(individual)):
            if random.random() < mutation_prob:
                node_id = node_list[i]

                # Option 1: Assign based on majority of neighbor zones
                if node_id in graph:
                    neighbors = list(graph.neighbors(node_id))
                    if neighbors:
                        neighbor_indices = [node_list.index(n) for n in neighbors if n in node_list]
                        neighbor_zones = [individual[j] for j in neighbor_indices]

                        if neighbor_zones:
                            # Choose most common neighbor zone OR randomly mutate
                            if random.random() < 0.7:  # 70% chance: local consistency
                                majority_zone = max(set(neighbor_zones), key=neighbor_zones.count)
                                individual[i] = majority_zone
                            else:
                                individual[i] = random.randint(0, num_zones - 1)
                    else:
                        # No neighbors? Just mutate randomly
                        individual[i] = random.randint(0, num_zones - 1)

        return individual,

    toolbox.register("mutate", mutate_clustered, graph=graph, mutation_prob=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ---------------- Parallel Processing for Speedup ----------------
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # ---------------- Run Genetic Algorithm ----------------
    population = toolbox.population(n=100)  # Increase population size for diversity
    elite_size = 5  # Keep the best 5 individuals unchanged each generation

    progress_bar = st.progress(0)

    for generation in range(200):  # Increase number of generations
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        # Evaluate all individuals
        fits = list(toolbox.map(toolbox.evaluate, offspring))

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # Elitism: Keep the best individuals
        elites = tools.selBest(population, elite_size)

        # Select next generation from offspring + elites
        population[:] = tools.selBest(population + offspring, k=len(population) - elite_size) + elites

        # Adaptive mutation: Reduce mutation rate as the population stabilizes
        mutation_prob = max(0.1 - 0.0005 * generation, 0.02)  # Starts at 0.1, decreases but not below 0.02

        # Re-register mutation with the new probability
        toolbox.unregister("mutate")
        toolbox.register("mutate", mutate_clustered, graph=graph, mutation_prob=mutation_prob)

        # Update progress bar
        progress = (generation + 1) / 200
        progress_bar.progress(progress)
        time.sleep(0.05)  # Reduce delay for faster computation

    pool.close()
    pool.join()

    # Return the best zoning plan
    best_ind = tools.selBest(population, k=1)[0]
    return best_ind

def visualize_zoning(map_object, graph, zoning):
    """
    Visualizes zoning on a Folium map without adding the legend.
    """
    zone_colors = {
        0: "#1E90FF",  # Residential
        1: "#2ECC71",  # Park/Green Space
        2: "#F1C40F",  # Commercial
        3: "#E74C3C",  # Industrial
        4: "#9B59B6",  # Educational
    }

    zone_names = {
        0: "Residential",
        1: "Park / Green Space",
        2: "Commercial / Business",
        3: "Industrial",
        4: "Educational",
    }

    for node_id, zone in zip(graph.nodes, zoning):
        node_data = graph.nodes[node_id]
        if 'x' in node_data and 'y' in node_data:
            coords = (node_data['y'], node_data['x'])
            color = zone_colors.get(zone % 5, "#7D7D7D")
            zone_label = zone_names.get(zone % 5, "Unknown")

            folium.CircleMarker(
                location=coords,
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(f"<b>Zone:</b> {zone_label}", max_width=250),
            ).add_to(map_object)
        else:
            print(f"⚠️ Node {node_id} is missing coordinates.")


def display_zone_legend():
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.4); width: 300px;
                    color: white; text-align: left;">
            <h4 style="margin-top: 0; text-align: center;">🗂️ Zoning Legend</h4>
            <div style="line-height: 1.8; font-size: 16px;">
                <span style="color:#1E90FF;">●</span> Residential<br>
                <span style="color:#2ECC71;">●</span> Park / Green Space<br>
                <span style="color:#F1C40F;">●</span> Commercial / Business<br>
                <span style="color:#E74C3C;">●</span> Industrial<br>
                <span style="color:#9B59B6;">●</span> Educational<br>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



def evaluate_zone_balance(zoning, num_zones=8):
    """
    Evaluates how balanced the zoning distribution is.
    A higher score (closer to 1) means more balanced zones.
    """
    zone_counts = np.bincount(zoning, minlength=num_zones)  # Count occurrences of each zone
    max_count = max(zone_counts)
    min_count = min(zone_counts)

    if max_count == 0:  # Avoid division by zero
        return 0

    balance_score = min_count / max_count  # Closer to 1 means better balance
    return round(balance_score, 2)  # Rounded for readability

# ===========================
# 🌍 STREAMLIT APP - SELECT AREA
# ===========================
st.title("SMAPPER")

st.write("🔍 **Select a location by entering a place name AND drawing a rectangle on the map.**")
# Initialize OpenCage Geocoder
api_key = "6b2ae8a0441f4888a5eb33f24bcd3f5a"
geocoder = OpenCageGeocode(api_key)

# User input location
location_name = st.text_input("Enter a location:", value="Bangalore")

# Use OpenCage for geocoding
result = geocoder.geocode(location_name)
if result:
    center_coords = (result[0]['geometry']['lat'], result[0]['geometry']['lng'])
else:
    st.error("Could not find the location. Try another place.")

# Create a folium map
m = folium.Map(location=center_coords, zoom_start=12)

# Add drawing tools
from folium.plugins import Draw

draw = Draw(export=True, draw_options={"polygon": False, "circle": False, "marker": False})
draw.add_to(m)

# Display the map
map_data = st_folium(m, width=700, height=500)

# Extract bounding box
if map_data and "all_drawings" in map_data and map_data["all_drawings"]:
    first_drawing = map_data["all_drawings"][0]
    if "geometry" in first_drawing:
        coordinates = first_drawing["geometry"]["coordinates"][0]
        min_lon = round(min(coord[0] for coord in coordinates), 5)  # Round to 5 decimal places
        max_lon = round(max(coord[0] for coord in coordinates), 5)  # Round to 5 decimal places
        min_lat = round(min(coord[1] for coord in coordinates), 5)  # Round to 5 decimal places
        max_lat = round(max(coord[1] for coord in coordinates), 5)  # Round to 5 decimal places

        st.write(f"Selected Bounding Box: {min_lat}, {min_lon}, {max_lat}, {max_lon}")

        # Fetch OSM data
        bbox = box(min_lon, min_lat, max_lon, max_lat)
        bbox_tuple = (min_lon, min_lat, max_lon, max_lat)

        roads_final_data = ox.features_from_bbox(bbox_tuple, tags={"highway": True})

        # Fetch land use data with geometries
        land_use_data = ox.features_from_bbox(bbox_tuple, tags={"landuse": True})

        # Fetch water bodies with geometries
        water_bodies_data = ox.features_from_bbox(bbox_tuple, tags={"natural": ["water", "lake", "river"]})

        # Fetch places with geometries
        places_data = ox.features_from_bbox(bbox_tuple, tags={"place": True})


        # Validate bounding box size (not shown in the original code)
        def validate_bbox_size(bbox_tuple):
            min_lon, min_lat, max_lon, max_lat = bbox_tuple
            lat_distance = geodesic((max_lat, min_lon), (min_lat, min_lon)).km
            lon_distance = geodesic((min_lat, min_lon), (min_lat, max_lon)).km
            area_km2 = lat_distance * lon_distance  # Approximate area
            return area_km2 <= 1000  # Max area 50 km²

        if validate_bbox_size(bbox_tuple):
            roads, roads_data, land_use, water_bodies, places = fetch_osm_data(bbox_tuple)
        else:
            st.error("Please select a smaller area.")

        if roads is not None:
            st.success(" ✅ OSM Data retrieved successfully.")
            # Create the map for the bounding box
            map_object = create_map_from_bbox([min_lon, min_lat, max_lon, max_lat])

            # Capture and crop the map
            cropped_image_path = capture_map_as_image([min_lon, min_lat, max_lon, max_lat])

            st.image(cropped_image_path, caption="Cropped Map of Selected Area")

            # Filter and display Roads Data
            if isinstance(roads_data, pd.Series):
                roads_data = roads_data.to_frame()  # Convert Series to DataFrame

            # Ensure 'highway' exists before filtering
            if isinstance(roads_data, pd.DataFrame) and 'highway' in roads_data.columns:
                filtered_roads = roads_data[roads_data['highway'].notna() & (roads_data['highway'] != "")]
                st.write("### Filtered Roads Data:")
                st.write(filtered_roads)
            else:
                st.error("Column 'highway' not found in roads_data!")

            if isinstance(land_use, pd.Series):
                land_use = land_use.to_frame()

            if isinstance(land_use, pd.DataFrame) and 'landuse' in land_use.columns:
                filtered_land_use = land_use[land_use['landuse'].notna() & (land_use['landuse'] != "")]
                st.write("### Filtered Land Use Data:")
                st.write(filtered_land_use)

            if isinstance(water_bodies, pd.Series):
                water_bodies = water_bodies.to_frame()

            if isinstance(water_bodies, pd.DataFrame) and 'natural' in water_bodies.columns:
                filtered_water_bodies = water_bodies[water_bodies['natural'].notna() & (water_bodies['natural'] != "")]
                st.write("### Filtered Water Bodies Data:")
                st.write(filtered_water_bodies)

            if isinstance(places, pd.Series):
                places = places.to_frame()

            if isinstance(places, pd.DataFrame) and 'place' in places.columns:
                filtered_places = places[places['place'].notna() & (places['place'] != "")]
                st.write("### Filtered Places Data:")
                st.write(filtered_places)

            # Create graph from roads
            graph = create_graph_from_osm_data(roads_final_data, land_use_data, water_bodies_data, places_data)

            # ✅ Densify the graph to add synthetic nodes (segments=2–5 for moderate density)
            graph = densify_graph(graph, segments=2, min_edge_length=50)

            if graph:
                st.success("✅ Graph created successfully from roads data.")
                st.write("⚙️ **Running Zoning Optimization...**")

                # Define constraints (you can adjust these weights)
                constraints = {
                    "road_access": 2.0,
                    "green_space": 1.5,
                    "industrial_buffer": 3.0,
                    "water_proximity": 2.0,
                    "urban_placement": 1.0,
                    "mixed_use": 1.5,
                    "walkability": 1.0,
                    "amenities": 2.0,
                    "edu_near_commercial": 5.0,
                    "industrial_clustering": 1.5
                }

                num_zones = 5  # Change as needed

                # Construct osm_data dictionary
                osm_data = {
                    "roads": roads_final_data,  # OSM road network data
                    "land_use": land_use_data,  # Land use data from OSM
                    "water_bodies": water_bodies_data,  # Water bodies data
                    "places": places_data  # Place information (e.g., neighborhoods)
                }

                # Run genetic algorithm for zoning optimization
                best_solution = optimize_zoning(graph, num_zones, constraints, osm_data)

                st.write("✅ **Zoning Optimization Completed!**")
                st.write("Best Zoning Layout (zones):", list(best_solution))
                st.write(f"Fitness Score: {best_solution.fitness.values[0]:.2f}")

                visualize_zoning(map_object, graph, best_solution)

                # Evaluate zoning balance
                balance_score = evaluate_zone_balance(best_solution, num_zones)
                st.write(f"📊 **Zone Balance Score:** {balance_score}")

                st.write("### 🗺️ Zoning Visualization")
                st.components.v1.html(map_object._repr_html_(), height=600)

                display_zone_legend()

            else:
                st.error("❌ Failed to create the graph.")

        else:
            st.error("❌ Could not retrieve GIS data for the selected area.")
