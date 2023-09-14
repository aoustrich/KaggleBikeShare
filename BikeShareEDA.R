# 
# Bike Share EDA Code
# 

# Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)

# read in data
bike <-  vroom("./train.csv")

# change factor columns
bike$season <- as.factor(bike$season)
bike$weather <- as.factor(bike$weather)


# EDA
skimr::skim(bike)
DataExplorer::plot_missing(bike) # No missing rows
plot_intro(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)
# plot_histrogram(bike)
GGally::ggpairs(bike)

# important factors could be... temp, atemp, humidity

# ggplot()
plot3 <-  ggplot(data= bike, mapping=aes(x=windspeed, y=count)) +
  geom_point() +
  geom_smooth()+
  labs(title="Count by Windspeed",x='Windspeed',y='Count')
  
plot4 <- ggplot(data= bike, mapping=aes(x=humidity, y=count)) +
  geom_point() +
  geom_smooth()+
  labs(title="Count by Humidity",x='Humidity',y='Count')

# patchwork stuff
plot1 <- plot_intro(bike)
plot2 <- plot_bar(bike)

(plot1 + plot2) / (plot3 + plot4)

