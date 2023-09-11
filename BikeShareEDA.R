# 
# Bike Share EDA Code
# 

# Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)

# read in data
bike <-  vroom("./train.csv")

# change factor columns
bike$season <- as.factor(bike$season)
bike$weather <- as.factor(bike$weather)

# EDA
dplyr::glimpse(bike)

skimr::skim(bike)

plot_intro(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)
DataExplorer::plot_histrograms(bike)
DataExplorer::plot_missing(bike)
GGally::ggpairs(bike)
