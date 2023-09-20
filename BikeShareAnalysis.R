library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

# read in data
bike.train <-  vroom("./train.csv")
bike.test <- vroom("./test.csv")

## Data Cleaning

# Changing row where weather is type 4 to type 3
bike.train <- bike.train %>% 
  # mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  select(-c("casual","registered"))


## Data Expansion

# my_recipe <- recipe(count ~ . , data = bike.train) %>%  # Set formula and dateset
#   # mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
#   # step_mutate(weatherF = factor(weather, #make factor of weather
#   #                               levels = c(1,2,3),
#   #                               labels=c("Sunny","Light Precip","Heavy Precip"))) %>%
#   step_dummy(all_nominal_predictors()) %>%  # create dummy variables
#   step_zv(all_predictors())  # remove near-zero variance variables

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


prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bike.train)

# Linear Regression Model Fitting ----------------------
# Define model
model <- linear_reg() %>% #Type of model
  set_engine("lm") 

# set up workflow
bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(model) %>% 
  fit(data = bike.train)

# look at the fitted model
extract_fit_engine(bike_workflow) %>% 
  tidy()
extract_fit_engine(bike_workflow) %>% 
  summary()


# make predictions
bike_preds <- predict(bike_workflow, new_data = bike.test)

# remove negative predictions
bike_preds <- bike_preds %>% 
  mutate(.pred = ifelse(.pred < 1 , 1, .pred)) %>% #
  mutate(.pred = round(.pred, digits = 3))

# prep data for submission
out <- cbind(bike.test$datetime, bike_preds) %>% 
  rename(., "datetime"= "bike.test$datetime") %>% 
  rename(., "count" = .pred) %>% 
  mutate(., "datetime" = as.character((datetime)))

# write to .csv
vroom_write(out, file="BikeSubmission.csv",delim=',')

# Poisson Regression ------------------------------------------------------
# define model
pois_mod <- poisson_reg() %>%
  set_engine("glm") 

# run workflow
bike_pois_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pois_mod) %>%
fit(data = bike.train) 

# get predictions and prep for kaggle
bike_predictions <- predict(bike_pois_workflow,new_data=bike.test)  %>%
  bind_cols(., bike.test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(1, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 

# look at the fitted model
extract_fit_engine(bike_pois_workflow) %>% 
  tidy()

# write out data
vroom_write(bike_predictions, file="BikeSubmissionPois.csv",delim=',')


# Penalized Regression ----------------------------------------------------
# convert count to log scale and remove casual 
bike.train <-  vroom("./train.csv") 
bike.train.l <- bike.train %>% 
  # mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  select(-c("casual","registered")) %>% 
  mutate(count=log(count))


# redo recipe
my_recipe <- recipe(count~., data=bike.train.l) %>%
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

# set up model
penal_model <- linear_reg(penalty = 0, mixture = 0) %>% 
  set_engine('glmnet')

# set up workflow
penal_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(penal_model) %>% 
  fit(data = bike.train.l)

# predict values and prep for kaggle
bike_preds_penal <- predict(penal_wf,new_data=bike.test)  %>%
  mutate(.pred=exp(.pred)) %>% 
  bind_cols(., bike.test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(1, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 

# look at the fitted model
extract_fit_engine(penal_wf) %>% 
  tidy()

# write out data
vroom_write(bike_preds_penal, file="BikeSubmissionPenalized.csv",delim=',')


### Poisson Penalized
# set up model
penal_pois_model <- poisson_reg(penalty = 0, mixture = 0.5) %>% 
  set_engine('glm')

# set up workflow
penal_pois_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(penal_pois_model) %>% 
  fit(data = bike.train.l)

# predict values and prep for kaggle
bike_preds_penal_pois <- predict(penal_pois_wf,new_data=bike.test)  %>%
  mutate(.pred=exp(.pred)) %>% 
  bind_cols(., bike.test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(1, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 

# write out data
vroom_write(bike_preds_penal_pois, file="BikeSubmissionPenalizedPoisson.csv",delim=',')



