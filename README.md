# 🦌 Moose Observations in Maine

A data science project analyzing moose sightings in Maine using iNaturalist data and machine learning to predict the best locations for moose spotting.

## 📊 Project Overview

This project uses historical moose observation data from iNaturalist to:
- Analyze moose sighting patterns across Maine
- Create interactive maps of observation locations
- Predict optimal moose spotting locations for June 2026
- Identify nearest cities to predicted hotspots

## 🗂️ Files

- **`moose_datascrape.py`** - Data collection script using iNaturalist API
- **`moose_observations.ipynb`** - Jupyter notebook with data exploration and visualizations
- **`moose_predictions.py`** - Machine learning script for predicting moose locations
- **`moose_predictions_june_2026.png`** - Map showing predicted best moose spotting locations for June 2026


## 🔍 Key Features

### Data Analysis
- **1,136 moose observations** from Maine
- Temporal analysis showing observation trends over time
- Geographic distribution mapping with dark theme visualization

### Machine Learning Predictions
- Random Forest model trained on historical June observations
- Predicts probability of moose sightings across Maine
- Identifies top 10 locations with highest probability scores

### Location Intelligence
- Maps nearest Maine cities to each predicted hotspot
- Distance calculations to help plan moose-watching trips
- Includes major cities: Portland, Bangor, Augusta, Bar Harbor, and more

## 🎯 Top Predicted Locations (June 2026)

The model identifies the best moose spotting locations with:
- Geographic coordinates (latitude/longitude)
- Probability scores
- Nearest city references with distances

## 🗺️ Visualizations

- **Dark theme maps** with green observation points
- **Probability heatmaps** showing likelihood of moose sightings
- **City markers** for geographic reference
- **Red stars** highlighting top prediction locations

## 🛠️ Technologies Used

- **Python** - Data analysis and machine learning
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Random Forest modeling
- **GeoPandas** - Geographic data processing
- **Matplotlib** - Data visualization
- **iNaturalist API** - Observation data source

## 📈 Model Performance

The Random Forest model is trained on historical June observations to predict similar seasonal patterns for future years, focusing on geographic and temporal features.

## 🚀 Getting Started

1. Install required packages:
```bash
pip install pandas numpy scikit-learn geopandas matplotlib seaborn
```

2. Run the Jupyter notebook:
```bash
jupyter notebook "Moose Observations.ipynb"
```

3. Generate predictions:
```bash
python moose_predictions.py
```

## 📍 Planning Your Moose Trip

Use the generated predictions and city references to plan your moose-watching adventure in Maine. The model suggests the highest probability locations along with practical information about nearby towns for accommodation and supplies.

---

*Data sourced from iNaturalist community observations. Predictions are based on historical patterns and should be used as guidance for wildlife viewing opportunities.*