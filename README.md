# Project Proposal: Predicting Residential Water Well Production

There are as many 2 million water wells in California, ranging from industrial and agricultural to residential.
The yield of a well is dependent of a variety of factors, but the most important is the availability of groundwater at a given location.
There is currently no reliable method of predicting well yield. Expensive geologic surveys can be performed to assess the location of ground water, but these are not always reliable.
Here I propose to analyze data on wells drilled in California to understand the how well yields are distributed across the California geography and use this data to estimate well production for new wells drilled.


This analysis uses data from Well Completion Reports provided by the California Department of Water Resources, available here: https://data.ca.gov/dataset/well-completion-reports

### Well Production Varies Across Geography

Intuition tells us that geography is essential to the availability of groundwater.   
<p float="left">
  <img src="/Well_yield_plot.png" width="400" />
  <img src="/Sonoma_plot.png" width="400" /> 
</p>
This is easily confirmed by visualizing the distribution of residential well yields across California. Looking across the state, it is immediately clear that certain areas, such as the Central Valley, generally have higher production wells. This distribution is aligned with our intuition--the Central Valley, famous for it's agriculture, is known to have abundant and accessible groundwater for irrigation. 

Looking at a more local level (in this case Sonoma County), we also see that production varies on a more local scale. Again, the distribution matches our intuition. For those familiar with Sonoma County geography, higher yield wells cluster in the Santa Rosa Valley with lower production wells clustering in the coastal mountain range.

This exploratory visualization motivates building an initial model to predict well yields using yield data from already existing wells.

### Predicting New Well Production from Neighboring Wells

To develop an initial model, I have defined a Voronoi set from the existing well data. By rasterizing this data we can get rough predictions for well yields. 

<p float="left">
  <img src="/Voronoi_set_plot.png" width="410" />
  <img src="/Voronoi_plot.png" width="410" /> 
</p>

The density of wells drilled is highly variable across the state, which complicates assessing the accuracy of the model as a whole as some regions will have poor accuracy due to more local data. To assess accuracy across the whole model, I used five-fold cross validation and used root mean square error as a measure of fitness.

A natural model for this analysis is a nearest neighbor interpolation. Here I have used the five nearest wells to predict well yields across the state. Weighting the neighboring wells by distance provides a slight improvement to the predictions.
<p float="left">
  <img src="/Nn_interp_plot.png" width="400" />
  <img src="/Wnn_interp_plot.png" width="400" /> 
</p>
### Taking this further...
In this initial analysis, I demonstrate that data from local residential wells alone can provide a rough model to predict well yield at a given location. 

In continuing this project, additional data from this set (e.g. depth drilled and well specifications) would further improve this model. From intuition and the exploratory visualizations in Sonoma County, it is also clear that geologic information has a profound effect on well yield. Data on local groundwater measurements, geologic and aquifer formations, precipitation is also available from the state of California and this information will be integrated into a better model. Additionally, further improvements can be made to the model by taking into consideration that some areas have very sparse data. Accounting for these areas separately using different methods could further improve the overall model.

In this analysis, I have only considered residential wells. Similar analyses will be performed on agricultral and industrial wells, which are much deeper and produce much higher yields, extending the use of this data to 

The deliverable from this project will be an interactive web app that allows a user to explore well production across California as well as enter a specific location and get a prediction of well yield along with information about the wells that exist around that location. This project will create a simple tool to allow users to instantly get an estimate of what they can expect if they dig a well, a service that is currently not available. This is invaluable information to someone who is looking at purchasing or building a house in rural areas and needs to assess the viability of access to water.






