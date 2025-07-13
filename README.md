# Moose in Maine

A data analysis project tracking moose observations in Maine using iNaturalist data.

## Project Overview

This project analyzes research-grade observations of moose (*Alces alces*) in Maine using data from the iNaturalist citizen science platform. The analysis includes:

- Retrieving moose observation data via the iNaturalist API
- Cleaning and processing observation records
- Visualizing observation locations and trends
- Analyzing seasonal patterns in moose sightings

## Data Sources

- Primary data comes from iNaturalist's API, focusing on research-grade observations
- Additional data loaded from CSV exports of iNaturalist observations

## Files in this Repository

- `moose_data.py`: Python module for retrieving moose observation data from iNaturalist
- `load_observations.py`: Script for loading and processing exported CSV observation data
- `Moose Observations.ipynb`: Jupyter notebook containing the main analysis and visualizations

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - matplotlib
  - pyinaturalist
  - jupyter

## Usage

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt` (if available)
3. Run the Jupyter notebook: `jupyter notebook "Moose Observations.ipynb"`

## Future Work

- Expand analysis to compare Maine moose populations with neighboring states
- Incorporate environmental data to analyze habitat preferences
- Create interactive maps of observation hotspots

## License

[Add your preferred license here]

## Contact

[Your contact information or GitHub username]
