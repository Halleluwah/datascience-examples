import pandas as pd
import plotly.graph_objects as go
import datetime as dt

# Prepare the data
eq = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv")

# Convert dates
eq["Date"] = pd.to_datetime(eq["Date"], errors = "coerce", format = "%m/%d/%Y")

min_year = int(eq["Date"].dt.year.min())

# Create figure
fig = go.Figure()

# Add traces
for i in range(min_year, int(eq["Date"].dt.year.max() + 1)):
  curr_year = eq[eq["Date"].dt.year == i]
  fig.add_trace(
      go.Densitymapbox(
        lat = curr_year["Latitude"],
        lon = curr_year["Longitude"],
        z = curr_year["Magnitude"], # values
        hoverinfo="z", # hover text
        colorscale="Plasma",
        colorbar_title="Magnitude of earthquakes",
        visible = True if i == min_year else "legendonly",
        radius = 15
      )
    )


# Create steps for the slider
steps = []
for i in range(len(fig.data)):
  step = dict(
    method = "update",
    args = [{"visible": [False] * len(fig.data)},
            {"title": "Year: " + str(min_year + i)}],
    label = str(min_year + i)
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
  title = "Year: {}".format(min_year),
  mapbox_style = "carto-darkmatter",
  mapbox_center_lon = 180,
  width = 1400,
  height = 900,
  sliders = sliders
)

fig.show()
