import pandas as pd
import plotly.express as px

# 1. Load your data
df = pd.read_csv('kerala_datasetv2.csv')

# 2. Create an interactive line plot
fig = px.line(df, x='timestamp', y='P', title='Time Series Data')

# 3. Enable a "rangeslider" (optional, but great for time series)
fig.update_xaxes(rangeslider_visible=True)

# 4. Save the plot as HTML instead of showing it
fig.write_html('microgrid_plot.html')
print("Plot saved to microgrid_plot.html")