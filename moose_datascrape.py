# Use the iNaturalist API with correct IDs and pull all relevant columns
import requests
from pyinaturalist import get_observations
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("Attempting to connect to iNaturalist API with correct IDs...")

try:
    # Get moose observations from Maine using the correct IDs
    observations = get_observations(
        taxon_id=522193,  # Moose (Alces alces)
        place_id=17,      # Maine
        quality_grade='research',
        per_page=200,
        pages=20,
        order_by='observed_on',
        order='desc'
    )
    
    print(f"Successfully retrieved {len(observations['results'])} observations!")
    
    # Convert to DataFrame with all relevant columns
    obs_data = [] 
    for obs in observations['results']:
        # Handle coordinates - they should be in the geojson or location field
        lat, lon = None, None
        
        # Try geojson coordinates first (most reliable)
        if obs.get('geojson', {}).get('coordinates'):
            coords = obs['geojson']['coordinates']
            lon, lat = float(coords[0]), float(coords[1])  # GeoJSON is [lon, lat]
        # Try location field as backup
        elif obs.get('location'):
            if isinstance(obs['location'], str):
                lat, lon = obs['location'].split(',')
                lat, lon = float(lat), float(lon)
        
        # Only add if we have valid coordinates
        if lat is not None and lon is not None:
            obs_data.append({
                'id': obs.get('id'),
                'uuid': obs.get('uuid'),
                'observed_on_string': obs.get('observed_on_string'),
                'observed_on': obs.get('observed_on'),
                'time_observed_at': obs.get('time_observed_at'),
                'time_zone': obs.get('time_zone'),
                'created_at': obs.get('created_at'),
                'updated_at': obs.get('updated_at'),
                'quality_grade': obs.get('quality_grade'),
                'tag_list': obs.get('tag_list', []),
                'description': obs.get('description'),
                'num_identification_agreements': obs.get('num_identification_agreements'),
                'num_identification_disagreements': obs.get('num_identification_disagreements'),
                'captive_cultivated': obs.get('captive_cultivated'),
                'latitude': lat,
                'longitude': lon,
                'positional_accuracy': obs.get('positional_accuracy'),
                'species_guess': obs.get('species_guess'),
                'scientific_name': obs.get('taxon', {}).get('name'),
                'common_name': obs.get('taxon', {}).get('preferred_common_name'),
                'iconic_taxon_name': obs.get('taxon', {}).get('iconic_taxon_name'),
                'taxon_id': obs.get('taxon', {}).get('id')
            })
    
    # Create DataFrame from obs_data
    df = pd.DataFrame(obs_data)
    print(f"Successfully created DataFrame with {len(df)} observations!")
    print(f"Columns: {df.columns.tolist()}")
    
except Exception as e  :
    print(f"Error with pyinaturalist: {e}")
  
# Convert to GeoDataFrame and display summary
if 'df' in locals() and len(df) > 0:
    # Convert observed_on to datetime first
    df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce', utc=True)
    
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    print(f"\nDataFrame Summary:")
    print(f"Shape: {df.shape}")
    
    # Check for valid dates
    valid_dates = df['observed_on'].dropna()
    if len(valid_dates) > 0:
        print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
    else:
        print("No valid dates found")
    
    print(f"Missing dates: {df['observed_on'].isnull().sum()}")
    
    print("\nFirst 3 observations:")
    print(df[['id', 'observed_on', 'latitude', 'longitude', 'quality_grade', 'description']].head(3))
    
    # Show data types
    print(f"\nData types:")
    print(df.dtypes)