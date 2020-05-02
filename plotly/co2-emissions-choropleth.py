# pip install pycountry

import pandas as pd
import plotly.graph_objects as go
import numpy as np

import pycountry

# Used to get ISO codes of countries in the DataFrame
def get_alpha_3(name):
  country = pycountry.countries.get(name = name)
  return country.alpha_3 if country else np.nan

# Prepare the data
co2 = pd.read_csv("https://raw.githubusercontent.com/Halleluwah/datasets/master/co2_emissions_tonnes_per_person.csv")

# Get ISO code of countries
co2["iso_alpha"] = co2["country"].apply(get_alpha_3)

# Drop rows of countries that were not recognised
co2 = co2[co2["iso_alpha"].notna()]

# Technically the dataset shows data from the year 1800, but only for a few countries
years = (1950, 2014)

# Create figure
fig = go.Figure()

# Add traces
for i in range(*years):
  fig.add_trace(
    go.Choropleth(
      locations = co2["iso_alpha"],
      z = co2[str(i)], # values
      text = co2["country"], # hover text
      hoverinfo = "text+z", # text to show on hover
      marker_line_color = "white", # line markers between states
      colorscale = "Plasma",
      colorbar_title = "CO2 emissions (tonnes per person)",
      visible = True if i == years[0] else "legendonly" # show only the colobar of the active year
    )
  )

# Create steps for the slider
steps = []
for i in range(len(fig.data)):
  step = dict(
    method = "update",
    args = [{"visible": [False] * len(fig.data)},
            {"title": "Year: " + str(years[0] + i)}],
    label = str(years[0] + i)
  )
  step["args"][0]["visible"][i] = True # Set i-th trace to "visible"
  steps.append(step)

# Create the slider
sliders = [dict(
  active = 0,
  currentvalue = {"visible": False},
  steps = steps,
  pad = { "r": 20, "l": 20, "b": 20 } # padding
)]

# Add the slider to the figure
fig.update_layout(
  title = "Year: {}".format(years[0]),
  width = 1400,
  height = 900,
  sliders = sliders
)

fig.show()
