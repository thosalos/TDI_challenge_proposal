# Project Proposal: 
# Waterworld: A Tool for Homeowners and Businesses to Improve Water Well Drilling Outcomes

In California, more than two million water wells have been drilled to service industrial, agricultural and residential water needs. Drilling a well is an expensive endeavor, costing between $15-25k for a residential well to $100k+ for high-production agricultural wells, and finding a reliable water supply is not guaranteed. Existing methods for predicting well yields range from the use of quack ‘water diviners’ to expensive geologic studies with questionable accuracy. Homeowners and businesses in California should be empowered with data to help improve well drilling outcomes. For my project, I propose building an interactive web app that allows a user to explore well production across California as well as to enter a specific location and get:
 
* Predictions about well yield
* Recommendations around ideal drilling depth
* Suggestions for ideal well drilling companies in the area with highest success rates given dynamic input (i.e. residential vs. agricultural wells)

Following is a preliminary feasibility study to analyze data on residential wells drilled in California to estimate well production for new wells drilled.

This analysis uses data from Well Completion Reports provided by the California Department of Water Resources, available here: https://data.ca.gov/dataset/well-completion-reports

### Well Production Varies Across Geography

Intuition tells us that geography is essential to the availability of groundwater.   
<p float="left">
  <img src="/Well_yield_plot.png" width="400" />
  <img src="/Sonoma_plot.png" width="400" /> 
</p>
This is easily confirmed by visualizing the distribution of residential well yields across California. Looking across the state, it is immediately clear that certain areas, such as the Central Valley, generally have higher production wells. This distribution is aligned with our intuition--the Central Valley, famous for agriculture, is known to have abundant and accessible groundwater for irrigation. 

Looking at a more local level (in this case Sonoma County), we also see that production varies on a more local scale. Again, the distribution matches our intuition. For those familiar with Sonoma County geography, higher yield wells cluster in the Santa Rosa Valley with lower production wells clustering in the coastal mountain range.

This exploratory visualization motivates building an initial model to predict well yields using data from already existing wells.

### Predicting New Well Production from Neighboring Wells

To develop an initial model, I have defined a Voronoi set from the existing well data. By rasterizing this data we can get rough predictions for well yields. 

<p float="left">
  <img src="/Voronoi_set_plot.png" width="410" />
  <img src="/Voronoi_plot.png" width="410" /> 
</p>

A natural model for this analysis is a nearest neighbor interpolation. Here I have used the five nearest wells to predict well yields across the state. Weighting the neighboring wells by distance provides a slight improvement to the predictions.
<p float="left">
  <img src="/Nn_interp_plot.png" width="410" />
  <img src="/Wnn_interp_plot.png" width="410" /> 
</p>

### Spatial models and additional data increase prediction accuracy by 30%

Integrating well specification data (well depth and casing length) and elevation data (from USGS) in more complex models can improve the predictions. Currently, I have tried random forests for spatial analysis, kriging methods and a spatial Durbin error model to improve the fit. The spatial Durbin error model provided the best fit as assessed by five-fold cross-validation of RMSE. The spatial lag model provided ~30% improvement compared to the null model.

### Visualization of model predictions provides essential validation
![alt text](![alt text](https://github.com/thosalos/TDI_challenge_proposal/blob/master/russian_river.png))

A visualization of the location of interest in Guerneville CA demonstrates the utility of geographic visualization in combination with the model. My model predicts ~30 GPM yield for my location. The visualization shows that this is skewed by several nearby high production wells that are in a river basin. By comparing the potential well of interest to wells that are on more similar terrain nearby, it can be determined that the expected yield will be lower.

### Taking this further

In this initial analysis, I demonstrate that data from local residential wells alone can provide a rough model to predict well yield at a given location. 

To further this project I would:
  
* Improve the model developed here by incorporating additional data around groundwater measurements, geologic information and precipitation
* Extend this model to industrial and agricultural wells
* Develop a model to recommend well depth that integrates data using approaches such as spatial lag models, random forests or kriging methods
* Generate a recommendation system for local well drilling companies based on their success rate
* Integrate these analyses into a coherent tool for homeowners and businesses
 
I believe the output of my project would be of high interest to the the ~370k homeowners and the ~100k industrial and agricultural firms using wells in California. I hope to empower homeowners and businesses with such a tool.
