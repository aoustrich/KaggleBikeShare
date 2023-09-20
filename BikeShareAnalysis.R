library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

# read in data
bike.train <-  vroom("./train.csv")
bike.test <- vroom("./test.csv")

# Data Cleaning -----------------------------------------------------------

# Remove `casual` and `registered` from training set because they're not in test
bike.train <- bike.train %>% 
  select(-casual, -registered)

# Make separate dataset with log(count) - This was done after the fact and led to lower score
bike.train.l <- bike.train %>% 
  mutate(count=log(count))

# Data Expansion ----------------------------------------------------------
# We'll need two recipes (one for the log(count) and one for the regular count)
my_recipe <- recipe(count~., data=bike.train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# log(count) recipe
my_recipe.l <- recipe(count~., data=bike.train.l) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# prep and bake recipes
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bike.train)

prepped_recipe.l <- prep(my_recipe.l)
bake(prepped_recipe.l, new_data = bike.train.l)

# Linear Regression Model Fitting ----------------------
# Define model
model <- linear_reg() %>% #Type of model
  set_engine("lm") 

# Set up workflow
bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(model) %>% 
  fit(data = bike.train)

# Look at the fitted model
extract_fit_engine(bike_workflow) %>% 
  tidy()
extract_fit_engine(bike_workflow) %>% 
  summary()

# Make function to get predictions, prep for kaggle, and export the data
predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,new_data=bike.test)  %>%
    bind_cols(., bike.test) %>% 
    select(datetime, .pred) %>% 
    rename(count=.pred) %>% 
    mutate(count=pmax(1, count)) %>% 
    mutate(datetime=as.character(format(datetime)))
  
  vroom_write(x, file=fileName,delim=',')
}

# Make Predictions, Clean, and Export data
predict_export(bike_workflow, "BikeSubmission.csv")

# Poisson Regression ------------------------------------------------------
# define model
pois_mod <- poisson_reg() %>%
  set_engine("glm") 

# run workflow
bike_pois_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pois_mod) %>%
fit(data = bike.train) 

# look at the fitted model
extract_fit_engine(bike_pois_workflow) %>% 
  tidy()

# predict and export
predict_export(bike_pois_workflow,"BikeSubmissionPois.csv" )

# Penalized Regression ----------------------------------------------------
# set up model
penal_model <- linear_reg(penalty = 0, mixture = 0) %>% 
  set_engine('glmnet')

# set up workflow using log transformed training set and log recipe
penal_wf <- workflow() %>% 
  add_recipe(my_recipe.l) %>% 
  add_model(penal_model) %>% 
  fit(data = bike.train.l)

# looked at model
extract_fit_engine(penal_wf) %>%
  tidy()

# predict and export
predict_export(penal_wf,"BikeSubmissionPenalized.csv")

# Penalized Poisson Regression --------------------------------------------
# set up model
penal_pois_model <- poisson_reg(penalty = 0, mixture = 0.5) %>% 
  set_engine('glm')

# set up workflow
penal_pois_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(penal_pois_model) %>% 
  fit(data = bike.train)

# predict values and prep for kaggle
predict_export(penal_pois_wf, "BikeSubmissionPenalizedPoisson.csv")