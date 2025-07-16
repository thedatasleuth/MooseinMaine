import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

# Load the observations data
file_path = "/Users/jessicachace/Downloads/observations-594739.csv/observations-594739.csv"
df = pd.read_csv(file_path)

# Create Maine boundary
maine_coords = [(-71.08, 43.09), (-71.08, 47.46), (-66.95, 47.46), (-66.95, 43.09), (-71.08, 43.09)]
maine = gpd.GeoDataFrame({'geometry': [Polygon(maine_coords)]}, crs="EPSG:4326")

# Prepare the data
df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude', 'observed_on'])

# Create features
df['month'] = df['observed_on'].dt.month
df['x'] = df['longitude']
df['y'] = df['latitude']

# Filter for June observations (month 6) to train on similar conditions
june_data = df[df['month'] == 6].copy()
june_data['moose_present'] = 1

# Generate random points as negative examples
def generate_random_points(n_points, bounds):
    minx, miny, maxx, maxy = bounds
    points = []
    while len(points) < n_points:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if maine.contains(Point(x, y)).any():
            points.append({'x': x, 'y': y, 'month': 6, 'moose_present': 0})
    return pd.DataFrame(points)

# Create training data
random_points = generate_random_points(len(june_data), maine.total_bounds)
training_data = pd.concat([
    june_data[['x', 'y', 'month', 'moose_present']],
    random_points
])

# Train the model
X = training_data[['x', 'y', 'month']]
y = training_data['moose_present']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Create prediction grid for June
grid_size = 0.02
x_range = np.arange(maine.total_bounds[0], maine.total_bounds[2], grid_size)
y_range = np.arange(maine.total_bounds[1], maine.total_bounds[3], grid_size)
xx, yy = np.meshgrid(x_range, y_range)

# Predict for June (month 6)
grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, 6)]
grid_scaled = scaler.transform(grid_points)
probabilities = rf.predict_proba(grid_scaled)[:, 1]

# Create results
results = pd.DataFrame({
    'longitude': xx.ravel(),
    'latitude': yy.ravel(),
    'probability': probabilities
})

# Filter to Maine only
geometry = [Point(x, y) for x, y in zip(results['longitude'], results['latitude'])]
results_gdf = gpd.GeoDataFrame(results, geometry=geometry, crs="EPSG:4326")
results_maine = gpd.clip(results_gdf, maine)

# Find top 10 best spots
top_spots = results_maine.nlargest(10, 'probability')

# Define major Maine cities and towns
maine_cities = {
    'Portland': (-70.2553, 43.6591),
    'Bangor': (-68.7712, 44.8016),
    'Augusta': (-69.7795, 44.3106),
    'Lewiston': (-70.1495, 44.1009),
    'South Portland': (-70.2409, 43.6415),
    'Auburn': (-70.2311, 44.0979),
    'Biddeford': (-70.4533, 43.4925),
    'Sanford': (-70.7739, 43.4395),
    'Saco': (-70.4428, 43.5009),
    'Westbrook': (-70.3712, 43.6770),
    'Waterville': (-69.6317, 44.5521),
    'Presque Isle': (-68.0158, 46.6781),
    'Caribou': (-68.0158, 46.8628),
    'Bar Harbor': (-68.2039, 44.3876),
    'Calais': (-67.2425, 45.1739),
    'Ellsworth': (-68.4197, 44.5434),
    'Machias': (-67.4615, 44.7145),
    'Millinocket': (-68.7097, 45.6578),
    'Rumford': (-70.5428, 44.5478),
    'Farmington': (-70.1533, 44.6698)
}

# Function to find nearest city
def find_nearest_city(lat, lon, cities_dict):
    min_distance = float('inf')
    nearest_city = None
    
    for city, (city_lon, city_lat) in cities_dict.items():
        distance = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_city = city
    
    # Convert to approximate miles (rough conversion)
    distance_miles = min_distance * 69  # 1 degree ≈ 69 miles
    return nearest_city, distance_miles

# Enhanced output with nearest cities
print("🦌 TOP 10 MOOSE SPOTTING LOCATIONS FOR JUNE 2026:")
print("=" * 70)
for i, spot in top_spots.iterrows():
    nearest_city, distance = find_nearest_city(spot.latitude, spot.longitude, maine_cities)
    rank = len(top_spots) - list(top_spots.index).index(i)
    print(f"{rank}. Lat: {spot.latitude:.3f}, Lon: {spot.longitude:.3f} "
          f"(Probability: {spot.probability:.1%})")
    print(f"   📍 Nearest city: {nearest_city} ({distance:.1f} miles away)")
    print()

# Create visualization
fig, ax = plt.subplots(figsize=(12, 10))

# Plot probability heatmap
results_maine.plot(column='probability', ax=ax, alpha=0.7, cmap='Greens', 
                   legend=True, legend_kwds={'label': "Moose Probability"})

# Plot Maine boundary
maine.boundary.plot(ax=ax, color='black', linewidth=2)

# Highlight top spots
top_spots.plot(ax=ax, color='red', markersize=100, alpha=0.8, marker='*')

# Add cities to the map
for city, (lon, lat) in maine_cities.items():
    ax.plot(lon, lat, 'bo', markersize=4, alpha=0.7)
    ax.text(lon, lat + 0.02, city, fontsize=8, ha='center', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

plt.title('Best Places to See Moose in Maine - June 2026', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('moose_predictions_june_2026.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📍 Map saved as 'moose_predictions_june_2026.png'")
print(f"🎯 Model trained on {len(june_data)} June observations")