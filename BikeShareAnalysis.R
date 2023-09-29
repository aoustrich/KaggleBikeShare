library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart) #used for regression trees
library(parallel)
library(ranger) # random forests

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
  # step_mutate(hour_cat = case_when( hour(datetime) < 6 ~ "Early",
  #                        hour(datetime) >= 6 & hour(datetime) <= 9 ~ "commute",
  #                        hour(datetime) > 9 & hour(datetime) < 16 ~ "day",
  #                        hour(datetime) >= 16 & hour(datetime) < 20 ~ "commute back",
  #                        hour(datetime) >= 20 ~ "night")) %>% 
  # step_mutate(hour_cat =factor(hour_cat, levels = 1:5, labels = c("Early","commute","day","commute back","night"))) %>% 
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
  # step_mutate(hour_cat = case_when( hour(datetime) < 6 ~ "Early",
  #                                   hour(datetime) >= 6 & hour(datetime) <= 9 ~ "commute",
  #                                   hour(datetime) > 9 & hour(datetime) < 16 ~ "day",
  #                                   hour(datetime) >= 16 & hour(datetime) < 20 ~ "commute back",
  #                                   hour(datetime) >= 20 ~ "night")) %>% 
  # step_mutate(hour_cat =factor(hour_cat, levels = 1:5, labels = c("Early","commute","day","commute back","night"))) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# prep and bake recipes
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bike.train)

prepped_recipe.l <- prep(my_recipe.l)
bake(prepped_recipe.l, new_data = bike.train.l)

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

# Make function to get  LOG predictions, prep for kaggle, and export the data
predict_export.l <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,new_data=bike.test)  %>%
    bind_cols(., bike.test) %>% 
    select(datetime, .pred) %>% 
    mutate(.pred=exp(.pred)) %>%
    rename(count=.pred) %>% 
    mutate(count=pmax(1, count)) %>% 
    mutate(datetime=as.character(format(datetime)))
  
  vroom_write(x, file=fileName,delim=',')
}

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
predict_export.l(penal_wf,"BikeSubmissionPenalized.csv")

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



# Tuning Penalized Regression ---------------------------------------------

# set up model
penal_model.tune <- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine('glmnet')

# set up workflow using log transformed training set and log recipe
penal_wf.tune <- workflow() %>% 
  add_recipe(my_recipe.l) %>% 
  add_model(penal_model.tune)

# create tuning grid
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)

# split data for cross validation
folds <- vfold_cv(bike.train.l, v = 20, repeats=1)

# run cross validation
cv_results <- penal_wf.tune %>% 
  tune_grid(resamples =folds,
            grid = tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

# plot results
collect_metrics(cv_results) %>%
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# select best model
best_tune <- cv_results %>% 
  select_best("rmse")

# finalize workflow
final_wf <- 
  penal_wf.tune %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=bike.train.l)

# predict and export
predict_export.l(final_wf,"BikeSubmissionPenalizedTune.csv")


# Regression Tree ---------------------------------------------------------
# set up model
treeModel <- decision_tree(tree_depth = tune(),
                           cost_complexity = tune(),
                           min_n = tune()) %>% 
            set_engine("rpart") %>% 
            set_mode("regression")

# set up workflow
treeWF <- workflow() %>% 
  add_recipe(my_recipe.l) %>%
  add_model(treeModel)

# create tuning grid
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

# split data for cross validation
folds <- vfold_cv(bike.train.l, v = 5, repeats=1)

    # levels=5, v=5 -> .48973
    # levels=10, v=10 -> more than 7 minutes....

# run cross validation
treeCVResults <- treeWF %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics=metric_set(rmse))


# select best model
best_tuneTree <- treeCVResults %>% 
  select_best("rmse")

# finalize workflow
finalTreeWF <- 
  treeWF %>% 
  finalize_workflow(best_tuneTree) %>% 
  fit(data=bike.train.l)

# predict and export
predict_export.l(finalTreeWF,"BikeSubmissionRegressionTree2.csv")

# Random Forest -----------------------------------------------------------
randForestModel <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
                  set_engine("ranger") %>% # What R function to use
                  set_mode("regression")

forestWF <- workflow() %>% 
  add_recipe(my_recipe.l) %>%
  add_model(randForestModel)

# create tuning grid
forest_tuning_grid <- grid_regular(mtry(range = c(1,10) ),
                            min_n(),
                            levels = 5)

# split data for cross validation
rfolds <- vfold_cv(bike.train.l, v = 5, repeats=1)

# run cross validation
treeCVResults <- forestWF %>% 
  tune_grid(resamples = rfolds,
            grid = forest_tuning_grid,
            metrics=metric_set(rmse)) #8.5 minute run time

# select best model
best_tuneForest <- treeCVResults %>% 
  select_best("rmse")

# finalize workflow
finalForestWF <- 
  forestWF %>% 
  finalize_workflow(best_tuneForest) %>% 
  fit(data=bike.train.l)

# predict and export
predict_export.l(finalForestWF,"BikeSubmissionRandomForest.csv")
