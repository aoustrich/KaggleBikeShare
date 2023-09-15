library(tidyverse)
library(tidymodels)
library(vroom)

# read in data
bike.train <-  vroom("./train.csv")
bike.test <- vroom("./test.csv")


## Data Cleaning

# Changing row where weather is type 4 to type 3
bike.train <- bike.train %>% 
  # mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  select(-c("casual","registered"))


## Data Expansion

my_recipe <- recipe(count ~ . , data = bike.train) %>%  # Set formula and dateset
  # mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  # step_mutate(weatherF = factor(weather, #make factor of weather
  #                               levels = c(1,2,3),
  #                               labels=c("Sunny","Light Precip","Heavy Precip"))) %>%
  step_dummy(all_nominal_predictors()) %>%  # create dummy variables
  step_zv(all_predictors())  # remove near-zero variance variables


prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bike.train)

## Model Fitting
model <- linear_reg() %>% #Type of model
  set_engine("lm") 

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(model) %>% 
  fit(data = bike.train)

# make predictions
bike_preds <- predict(bike_workflow, new_data = bike.test)

# remove negative predictions
bike_preds <- bike_preds %>% 
  mutate(.pred = ifelse(.pred < 0 , 0, .pred)) %>% 
  mutate(.pred = round(.pred, digits = 3))

# prep data for submission
out <- cbind(bike.test$datetime, bike_preds) %>% 
  rename(., "datetime"= "bike.test$datetime") %>% 
  rename(., "count" = .pred) %>% 
  mutate(., "datetime" = as.character((datetime)))

# write to .csv
vroom_write(out, file="BikeSubmission.csv",delim=',')
