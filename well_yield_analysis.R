
devtools::install_github('rspatial/rspatial')

library(readr)
library(dplyr)
library(ggplot2)
library(maps)
library(ggmap)
library(randomForest)
library(raster)
library(sp)
library(rgdal)
library(rspatial)
library(dismo)
library(gstat)
library(spdep)
library(spatialreg)
library(elevatr)

#wells <- read_csv("OSWCR.csv")

# Clean data for useful fields; remove outliers and incomplete data.

res_wells <- filter(wells, PlannedUseFormerUse == "Water Supply Domestic") #~367K entries

res_wells_with_yields <- filter(res_wells, WellYield != "None", WellYield >= 0) 
res_wells_with_yields$WellYield <- as.numeric(res_wells_with_yields$WellYield)
res_wells_with_yields$BottomofPerforatedInterval <- as.numeric(res_wells_with_yields$BottomofPerforatedInterval)
res_wells_with_yields$TopOfPerforatedInterval <- as.numeric(res_wells_with_yields$TopOfPerforatedInterval)


#69 entries use archaic "miner's inches measurement"
res_wells_gpm <- filter(res_wells_with_yields, WellYieldUnitofMeasure == "GPM") 

#Remove outliers
yield_outliers <- boxplot(res_wells_gpm$WellYield, plot=FALSE)$out
clean_res_wells <- res_wells_gpm[-which(res_wells_gpm$WellYield %in% yield_outliers),]

depth_outliers <- boxplot(clean_res_wells$TotalCompletedDepth, plot=FALSE)$out
clean_res_wells <- clean_res_wells[-which(clean_res_wells$TotalCompletedDepth %in% depth_outliers),]
clean_res_wells <- dplyr::select(clean_res_wells, CountyName, DecimalLatitude, DecimalLongitude, TotalCompletedDepth, TopOfPerforatedInterval, BottomofPerforatedInterval, WellYield)
clean_res_wells <- na.omit(clean_res_wells) # ~108K complete entries

clean_res_wells$casing_length <- clean_res_wells$BottomofPerforatedInterval - clean_res_wells$TopOfPerforatedInterval
clean_res_wells <- filter(clean_res_wells, casing_length >= 0) 


# exploratory visualization of data

well_depth_county <- ggplot(clean_res_wells) +
  geom_boxplot(aes(x = CountyName, y = TotalCompletedDepth)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("County") +
  ylab("Well Depth (ft)") +
  ggtitle("Depths of Domestic Wells Drilled in California")

well_yield_county <- ggplot(clean_res_wells) +
  geom_boxplot(aes(x = CountyName, y = WellYield)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("County") +
  ylab("Well Yield (GPM)") +
  ggtitle("Yields of Domestic Wells Drilled in California")

depth_by_yield <- ggplot(clean_res_wells) +
  geom_point(aes(TotalCompletedDepth, WellYield)) +
  xlab("Well Depth (ft)") +
  ylab("Well Yield (GPM)") +
  ggtitle("Yields of Domestic Wells by Depth")

well_yield_county
well_depth_county
depth_by_yield

# Plot spatial information about domestic wells in California

# Get map data
states <- map_data("state")
cali <- subset(states, region == "california")
counties <- map_data("county")
ca_county <- subset(counties, region == "california")

# Base plot of california with counties
ca_base <- ggplot(data = cali, mapping = aes(x = long, y = lat, group = group)) +
  coord_fixed(1.3) +
  geom_polygon(color = "black", fill = "gray") +
  geom_polygon(data = ca_county, fill = NA, color = "white") +
  geom_polygon(color = "black", fill = NA) +
  theme_nothing()

# Calculate Number of Wells Per County and merge with map data
cali_wells <- dplyr::select(clean_res_wells, CountyName, WellYield, DecimalLatitude, DecimalLongitude)
well_counts <- cali_wells %>% count(CountyName)
well_counts <- rename(well_counts, subregion = CountyName, wells = n)
well_counts$subregion <- tolower(well_counts$subregion)
ca_county_counts <- inner_join(ca_county, well_counts, by = "subregion")

no_axes <- theme(
  axis.text = element_blank(),
  axis.line = element_blank(),
  axis.ticks = element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank(),
  axis.title = element_blank()
)

# Plot number of wells per county
wells_per_county <- ca_base +
  geom_polygon(data = ca_county_counts, aes(fill = wells), color = "white") +
  geom_polygon(color = "black", fill = NA) +
  theme_bw() +
  no_axes +
  theme(legend.title = element_blank()) +
  ggtitle("Domestic Wells in California Counties")

wells_per_county


# Well yields across California
well_yields <- cali_wells %>%
  group_by(CountyName) %>%
  summarize(median_yield = median(WellYield, na.rm = TRUE))
well_yields <- rename(well_yields, subregion = CountyName)
well_yields$subregion <- tolower(well_yields$subregion)

ca_county_yields <- inner_join(ca_county, well_yields, by = "subregion")

yields_per_county <- ca_base + 
  geom_polygon(data = ca_county_yields, aes(fill = median_yield), color = "white") +
  geom_polygon(color = "black", fill = NA) +
  theme_bw() +
  no_axes +
  theme(legend.title = element_blank()) +
  ggtitle("Median Domestic Well Production in California Counties")

yields_per_county


# Distribution of Well Production in California Counties

quantile(cali_wells$WellYield, probs = c(.33, .66))
cali_wells$production <- cut(cali_wells$WellYield, 
                             breaks = c(-Inf, 12, 32, Inf),
                             labels = c("low", "medium", "high"))

production_counts <- cali_wells %>% group_by(CountyName) %>% 
  count(production) %>%
  group_by(CountyName) %>%
  mutate(freq = n/sum(n))

county_production <- ggplot(production_counts) +
  geom_bar(aes(x = CountyName, y = freq, fill = production), stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme(legend.title = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank()) +
  labs(title = "Domestic Well Production Varies by County")

county_production


# Local well production in the North Bay
ca_county <- subset(counties, region == "california")
sonoma <- subset(ca_county, subregion == "sonoma")
sonoma_wells <- subset(cali_wells, CountyName == "Sonoma")

sonoma_box <- make_bbox(lon = sonoma$long, lat = sonoma$lat, f = .1)

sonoma_map_data <- get_map(location = sonoma_box, source = "google", maptype = "terrain")

sonoma_map <- ggmap(sonoma_map_data) +
  geom_point(data = sonoma_wells, 
             mapping = aes(y = DecimalLatitude, x = DecimalLongitude, color = production),
             alpha = .8, cex = 0.8) +
  no_axes +
  theme(legend.title = element_blank()) +
  ggtitle("Domestic Well Yield in Sonoma County Varies with Geography")

sonoma_map


# classify well production (Low, Medium, High) using raw Long/Lat and other predictors with a random forest

cali_features <- dplyr::select(clean_res_wells, DecimalLatitude, DecimalLongitude, TotalCompletedDepth, TopOfPerforatedInterval, BottomofPerforatedInterval, WellYield, casing_length)

cali_production <- cali_features
cali_production$production <- cut(cali_production$WellYield, 
                                  breaks = c(-Inf, 12, 32, Inf),
                                  labels = c("low", "medium", "high"))
cali_production <- dplyr::select(cali_production, -WellYield)


train <- sample(nrow(cali_production), 0.7*nrow(cali_production), replace = FALSE)
train_set <- cali_production[train,]
valid_set <- cali_production[-train,]

model <- randomForest(production ~ ., data = train_set, ntree = 500, mtry = 6, importance = TRUE)
model

# Predicting on train set
pred_train <- predict(model, train_set, type = "class")
# Checking classification accuracy
table(pred_train, train_set$production)  


# Predicting on Validation set
pred_valid <- predict(model, valid_set, type = "class")
# Checking classification accuracy
mean(pred_valid == valid_set$production)                    
table(pred_valid,valid_set$production)

importance(model)
varImpPlot(model)



# Develop a model that works better with location data
well_coords <- dplyr::select(cali_features, DecimalLongitude, DecimalLatitude)
well_sp <- SpatialPoints(well_coords, proj4string=CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"))
well_sp <- SpatialPointsDataFrame(well_sp, north_wells)
cali_sp <- sp_data("counties")
#define groups for mapping
cuts <- c(0, 10, 20, 50, 75, Inf)
# set up a palette of interpolated colors
blues <- colorRampPalette(c('yellow', 'orange', 'blue', 'dark blue'))
pols <- list("sp.polygons", cali_sp, fill = "lightgray")
yield_distribution <- spplot(well_sp, "WellYield", cuts=cuts, col.regions=blues(5), sp.layout=pols, pch=20, cex=0.5, 
                             colorkey = TRUE, main = "Distribution of Well Yields (GPM)")


# Voronoi set to define all regions between well data points and define by production

# Teales-Albert Coord System
# Transform lat/long to planar coordinates
TA <- CRS("+proj=aea +lat_1=32 +lat_2=42 +lat_0=0 +lon_0=-125 +x_0=0 +y_0=-4000000 +datum=NAD83 +units=m +ellps=GRS80 +towgs84=0,0,0")
well_ta <- spTransform(well_sp, TA)
cali_ta <- spTransform(cali_sp, TA)

well_voron <- voronoi(well_ta)
cali_agg <- aggregate(cali_ta)
cali_voron <- raster::intersect(well_voron, cali_agg)
voronoi_plot <- spplot(cali_voron, 'WellYield', col.regions=rev(get_col_regions()), lwd = 0.2,
                       main = "Voronoi Set of Well Yields")
voronoi_plot

# Rasterize Voronoi cells

r <- raster(cali_ta, res=1000)
vr <- rasterize(cali_voron, r, field = cali_voron$WellYield)
plot(vr, main = "Voronoi Predictions of Well Yields (GPM)", axes = FALSE)


# 5-fold cross-validation of Voronoi predictions using RMSE

RMSE <- function(observed, predicted) {
  sqrt(mean((predicted - observed)^2, na.rm=TRUE))
}

# Define the RMSE null
null <- RMSE(mean(well_sp$WellYield), well_sp$WellYield)

kf <- kfold(nrow(well_ta))
rmse <- rep(NA, 5)
for (k in 1:5) {
  test <- well_ta[kf == k, ]
  train <- well_ta[kf != k, ]
  v <- voronoi(train)
  p <- extract(v, test)
  rmse[k] <- RMSE(test$WellYield, p$WellYield)
}
rmse
mean(rmse)
1 - (mean(rmse) / null)

# Interpolate well production from the 5 nearest neighbors

gs <- gstat(formula=WellYield~1, locations=well_ta, nmax=5, set=list(idp = 0))
nn <- interpolate(r, gs)

nnmsk <- mask(nn, vr)
plot(nnmsk, main = "Nearest Neighbor Interpolation of Well Production (GPM)", axes = FALSE)

# Five-fold cross-validation of nearest neighbor interpolation

rmsenn <- rep(NA, 5)
for (k in 1:5) {
  test <- well_ta[kf == k, ]
  train <- well_ta[kf != k, ]
  gscv <- gstat(formula=WellYield~1, locations=train, nmax=5, set=list(idp = 0))
  p <- predict(gscv, test)$var1.pred
  rmsenn[k] <- RMSE(test$WellYield, p)
}

rmsenn
mean(rmsenn)
1 - (mean(rmsenn) / null)


# Weighted Nearest Neighbor Interpolation
gs <- gstat(formula=WellYield~1, locations=well_ta, nmax = 5)
idw <- interpolate(r, gs)
# inverse distance weighted interpolation
idwr <- mask(idw, vr)
plot(idwr, main = "Weight NN Interpolation of Well Production", axes = FALSE)


# Weighted Nearest Neighbor Five-fold cross-validation

rmse <- rep(NA, 5)
for (k in 1:5) {
  test <- well_ta[kf == k, ]
  train <- well_ta[kf != k, ]
  gs <- gstat(formula=WellYield~1, locations=train)
  p <- predict(gs, test)
  rmse[k] <- RMSE(test$WellYield, p$var1.pred)
}

rmse
mean(rmse)
1 - (mean(rmse) / null)


# Linear spatial Durbin error models
well_nn <- knearneigh(well_ta, k = 5)
wellnb <- knn2nb(well_nn)
listw1 <- nb2listw(wellnb)

reg.eq1 <- WellYield ~ TotalCompletedDepth + casing_length
reg1 <- lmSLX(reg.eq1, data = well_ta@data, listw = listw1)
summary(reg1)
lm.LMtests(reg1, listw1, test = "all")
impacts(reg1, listw = listw1)
summary(impacts(reg1, listw = listw1, R = 500, zstats = TRUE))

# Get point elevations and compute linear spatial Durbin error model
elevations <- get_elev_point(well_ta, src = "aws")
reg.eq2 <- WellYield ~ TotalCompletedDepth + casing_length + elevation
reg2 <- lmSLX(reg.eq2, data = elevations@data, listw = listw1)
lm.LMtests(reg2, listw1, test = "all")
summary(reg2)

# Just elevation in spatial error model
reg.eq3 <- WellYield ~ elevation
reg3 <- lmSLX(reg.eq3, data = elevations@data, listw = listw1)
summary(reg3)

#Get raster of elevations across the state
locations <- well_ta@bbox
locs <- data.frame(x = well_ta@bbox[1,], y = well_ta@bbox[2, ])
locs$x <- well_ta@bbox[1,]
locs$y <- well_ta@bbox[2]

# DEM for california for predicting values including elevation 
elev_rast <- get_elev_raster(locs, z = 1, prj = "+proj=aea +lat_1=32 +lat_2=42 +lat_0=0 +lon_0=-125 +x_0=0 +y_0=-4000000 +datum=NAD83 +units=m +ellps=GRS80 +towgs84=0,0,0")
plot(elev_rast)

# Random Forest model
elev_rast <- raster(elevations)
rf_coords <- as.data.frame(elevations@coords)
rf_data <- elevations@data
rf_data <- dplyr::select(rf_data, -DecimalLatitude, -DecimalLongitude, -elev_units)
task = makeRegrTask(data = rf_data, target = "WellYield", coordinates = rf_coords)
rf_learn = makeLearner(cl = "regr.ranger", predict.type = "response")

# Tune random forest hyperparameters
# spatial partitioning
perf_level = makeResampleDesc("SpCV", iters = 5)
# specifying random search
ctrl = makeTuneControlRandom(maxit = 50L)

ps = makeParamSet(
  makeIntegerParam("mtry", lower = 1, upper = ncol(rf_data) - 1),
  makeNumericParam("sample.fraction", lower = 0.2, upper = 0.9),
  makeIntegerParam("min.node.size", lower = 1, upper = 10)
)

tune = tuneParams(learner = rf_learn, 
                  task = task,
                  resampling = perf_level,
                  par.set = ps,
                  control = ctrl, 
                  measures = mlr::rmse)

rf_learn = makeLearner(cl = "regr.ranger",
                       predict.type = "response",
                       mtry = tune$x$mtry, 
                       sample.fraction = tune$x$sample.fraction,
                       min.node.size = tune$x$min.node.size)
rf_model = train(rf_learn, task)

# Plot just Guernville
local_map_data <- get_map(location = c(lon = -122.9769, lat = 38.5201), source = "google", maptype = "terrain", zoom = 12)
ggmap(north_map_data)

# Convert model prediction to polygons to map with ggmap
final_raster <- projectRaster(idwr, crs = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
raster_poly <- rasterToPolygons(final_raster)
raster_data <- as.data.frame(raster_poly)
raster_data$id <- 1:nrow(raster_data)
poly_plot <- polygons(raster_poly)
poly_plot$pred <- raster_data
fort <- fortify(poly_plot)
fort_merge <- merge(fort, raster_data, by.x = "id", by.y = "id")

# Specific location of interest -- extract prediction
gville_house <- data.frame(lat = 38.503876, lon = -123.007919, elev = 284.988)
gville_house <- SpatialPoints(matrix(c(-123.007919, 38.503876), nrow = 1, ncol = 2), proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs") )
extract(final_raster, gville_house)

# Plot local map
raster_map <- ggmap(local_map_data)
raster_map + 
  geom_polygon(data = fort_merge, 
               aes(x = long, y = lat, group = group, 
                   fill = fort_merge$var1.pred), 
               size = 0, 
               alpha = 0.5) +
  scale_fill_gradientn(colors = c('yellow', 'orange', 'blue', 'dark blue')) +
  geom_point(data = north_wells, 
             mapping = aes(y = DecimalLatitude, x = DecimalLongitude, color = production), cex = 2) +
  geom_point(data = gville_house, mapping = aes(x = lon, y = lat), cex = 4)
