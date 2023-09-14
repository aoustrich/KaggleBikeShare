library(tidyverse)
library(tidymodels)
library(vroom)

# read in data
bike.train <-  vroom("./train.csv")
bike.test <- vroom("./test.csv")


## Data Cleaning

# Changing row where weather is type 4 to type 3
bike.train <- bike.train %>% 
  mutate(weather = ifelse(weather == 4, 3, weather)) 

## Data Expansion

my_recipe <- recipe(count ~ . , data = bike.train) %>%  # Set formula and dateset
  step_mutate(weatherF = factor(weather, #make factor of weather
                                levels = c(1,2,3), 
                                labels=c("Sunny","Light Precip","Heavy Precip"))) %>% 
  step_dummy(all_nominal_predictors()) %>%  # create dummy variables
  step_zv(all_predictors()) # remove near-zero variance variables

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bike.train)



