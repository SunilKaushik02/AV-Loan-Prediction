# install libraries
libraries <- c('readxl','tidyverse','ggplot2','knitr','readr',
               'lubridate','RColorBrewer','ISLR','devtools','tidymodels',
               'scales','data.table','doParallel','textrecipes','glmnet','ranger','stacks',
               'xgboost','vip')

new_libs <- libraries[!(libraries %in% installed.packages()[,'Package'])]
new_libs

if (length(new_libs)){install.packages(libraries,dependencies = T)}

sapply(libraries,require,character.only=T)

doParallel::registerDoParallel(cores=8)

# load the file
data=read_excel('C:\\Learning\\Analytics Vidya\\Loan Prediction\\Train.xlsx',sheet='Train') 

holdout=read_csv('C:\\Learning\\Analytics Vidya\\Loan Prediction\\Test.csv') 

data$Loan_Status <- as.factor(ifelse(data$Loan_Status=='Y',1,0))

head(data)

summary(data)

# split data

set.seed(2)
spl <- initial_split(data,prop=0.8,strata = Loan_Status)
train <- training(spl)
test <- testing(spl)

train %>% count(Loan_Status) %>% 
  summarise(prop=n/sum(n))

test %>% count(Loan_Status) %>% 
  summarise(prop=n/sum(n))

# eda

# 1. look at which var might be potential predictors
# 2. try to look at diff scales to und whether it's normally distributed so that 
# linear model can be used

table(train$Loan_Status) #cant go worse than sd

train %>% ggplot(aes(LoanAmount))+geom_histogram()+scale_x_log10()

train %>% ggplot(aes(Loan_Amount_Term))+geom_histogram()+scale_x_log10()

train %>% ggplot(aes(Credit_History,LoanAmount,group=Credit_History)) + geom_boxplot()

train %>% ggplot(aes(ApplicantIncome,LoanAmount,group=ApplicantIncome)) + geom_point()+scale_x_log10()

train %>% ggplot(aes(ApplicantIncome,CoapplicantIncome)) + geom_point()+scale_x_log10()+scale_y_log10()

# looking at categoricadata

train %>% 
  group_by(Loan_Status,Gender,Married,Dependents,Education) %>% 
  count(sort=T)

train %>% 
  group_by(Loan_Status) %>% 
  count(Education,sort=T)

# train$Loan_Status[is.na(train$Loan_Status)] <- 0

# then modelling
# 1. start with linear model so tat ul understand d rship between models

mset <- metric_set(roc_auc)
train_fold <- train %>% vfold_cv(5,strata = Loan_Status)

lin_spec <- logistic_reg(penalty=0.001) %>% set_engine('glmnet')

lin_rec <- recipe(Loan_Status~Dependents+Self_Employed+Property_Area+ApplicantIncome+LoanAmount+Credit_History,data=train) %>% 
  step_impute_bag(all_predictors()) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_log(ApplicantIncome,LoanAmount,base=10,offset=1) %>% 
  step_interact(terms=~(ApplicantIncome+LoanAmount):Credit_History) %>% 
  step_zv(all_predictors()) %>% 
  step_rm(ApplicantIncome,LoanAmount)

# step_mutate(max_players=pmin(max_players,30)) %>% #bin size
# step_ns(year,deg_free = 5) %>% #non-linear splines only for year, use "tune()" if using tune_grid

# lin_rec %>%
#   prep() %>%
#   juice() %>%
#   View()

lin_wflow <- workflow() %>% 
  add_recipe(lin_rec) %>% 
  add_model(lin_spec) 

cv <- lin_wflow %>%
  fit_resamples(train_fold)
  # tune_grid(train_fold,
  #           grid=crossing(penalty=10^seq(-5,1,length.out=5)),
  #           metrics=mset,
  #           control=control_stack_grid())

cv %>% 
  collect_metrics()

cv %>%
  autoplot() #only for viewing tuning results n check if more tuning is required by looking at straight line

# fit

lin_fit <- lin_wflow %>% 
  fit(train)

tidy(lin_fit)
glance(lin_fit)

#predict

lin_pred <- predict(lin_fit,test,type='prob') %>% 
  bind_cols(predict(lin_fit,test))

yardstick::roc_auc(truth=test$Loan_Status,estimate=lin_pred$.pred_1)
accuracy(table(test$Loan_Status,lin_pred$.pred_class))

# results and iterations for linear reg

# better than .27 initially
# sec stage add splines for year - 0.219
# 3rd add mechanics - 0.212
# 4th add glmnet instead of lm in engine - 0.207
# 5th add designer - 1900k uniq n use tokenize
#6th add age - 0.210
# 7th adding categories - unite(cateories) - 0.201 nice :)

# next try tree models

# train_fold_rf <- train %>% vfold_cv(5)

rf_spec <- rand_forest("classification",
                       mtry=tune(),
                       trees=tune()) %>% 
  set_engine("ranger") %>% 
  set_args(importance='impurity')

rf_rec <- recipe(Loan_Status~Dependents+Self_Employed+Property_Area+ApplicantIncome+LoanAmount+Credit_History,data=train) %>% 
  step_impute_bag(all_predictors()) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_log(ApplicantIncome,LoanAmount,base=10,offset=1) %>% 
  step_interact(terms=~(ApplicantIncome+LoanAmount):Credit_History) 
  # step_rm(ApplicantIncome,LoanAmount)

# lin_rec %>%
#   prep() %>%
#   juice() %>%
#   View()

rf_wflow <- workflow() %>% 
  add_recipe(rf_rec) %>% 
  add_model(rf_spec) 

rf <- rf_wflow %>%
  tune_grid(train_fold,
            grid=crossing(mtry=1:3,
                          trees=c(500,700,1000)),
            # grid=crossing(deg_free=1:7),
            # grid=crossing(max_tokens=c(500,800,1000)),
            # grid=crossing(penalty=10^seq(-7,-1.5,1)),
            metrics=mset,
            control=control_stack_grid())

rf %>% 
  collect_metrics() %>% 
  arrange(mean)

rf %>%
  autoplot() #only for viewing tuning results n check if more tuning is required by looking at straight line

# results:
#   1. slowly changing mtry n trees from 2:6 then inc by 4's' 0.179 - look for plateaus in autoplot
# 2. inc to 10 fold from 5 fold after tuning - 0.174
# 3. add back categories n only 1st variable without adding category 1 - ****** important step

# checking vtest

rf_wflow %>% 
  finalize_workflow(list(trees=500,mtry=1)) %>% 
  last_fit(split=spl) %>% 
collect_metrics()

# ensembling rf + lin

rf_chosen <- rf %>% 
  filter_parameters(trees==500,mtry==1)

lin_chosen <-  cv %>% 
  filter_parameters(penalty==0.0001)

# lin_chosen <- lin_wflow %>% 
#   finalize_workflow(select_best(cv)) #another alternative

# checking score on test data
# lin_chosen %>% last_fit(split=spl) %>% 
#   collect_metrics()

# check abut imporatnce of var

rf_fit <- rf_wflow %>% 
  finalize_workflow(list(trees=500,mtry=7)) %>% 
  fit(train)

ranger::importance(rf_fit$fit$fit$fit)

# blended

ensemble <- stacks() %>% 
  add_candidates(lin_chosen) %>% 
  add_candidates(rf_chosen)

ensemble_blended <- ensemble %>% blend_predictions()

ensemble_fit <- ensemble_blended %>% 
  fit_members()

# to predict
# predict(ensemble_fit,test)

predict(ensemble_fit,test) %>% bind_cols(test) %>% 
  roc_auc(as.numeric(as.character(Loan_Status)),as.numeric(as.character(.pred_class))) #getting rmse for test data

predict(ensemble_fit,holdout) %>% bind_cols(holdout)  #getting rmse for final holdout data

# getting training on al data

ensemble_blended_full <- rf
ensemble_blended_full$train <- data

ensemble_fit_full <- ensemble_blended_full %>% fit_members()

# ensemble_fit_full %>% 
#   predict(data) %>% 
#   bind_cols(data) %>% 
#   rmse(geek_rating,.pred) #this will overfit d training data as is the case

attempt1 <- ensemble_fit_full %>% 
  predict(holdout) %>% 
  bind_cols(holdout %>% select(Loan_ID)) %>% 
  select(Loan_ID,Loan_Status=.pred_class)

attempt1$Loan_Status <- ifelse(attempt1$Loan_Status==1,'Y','N')

write_csv(attempt1,"C:/Learning/Analytics Vidya/Loan Prediction/attempt1.csv")

# attempt 1 was 0.79 - rank 886 not bad :)

# next try xgboost

train_fold_xg <- train %>% vfold_cv(10)

xg_spec <- boost_tree("classification",
                      mtry=tune(),
                      trees=tune(),
                      tree_depth=tune(),
                      learn_rate=tune()) %>% 
  set_engine("xgboost")

xg_rec <- recipe(Loan_Status~Dependents+Self_Employed+Property_Area+ApplicantIncome+LoanAmount+Credit_History,data=train) %>% 
  step_impute_bag(all_predictors()) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_log(ApplicantIncome,LoanAmount,base=10,offset=1) 

# lin_rec %>%
#   prep() %>%
#   juice() %>%
#   View()

xg_wflow <- workflow() %>% 
  add_recipe(xg_rec) %>% 
  add_model(xg_spec) 

xg <- xg_wflow %>%
  # fit_resamples(train_fold)
  tune_grid(train_fold_xg,
            grid=crossing(
              mtry=1:3,
              trees=c(700),
              tree_depth=1:3,
              learn_rate=c(0.05,0.01)), #low learning n more trees cud overfit!
            # grid=crossing(deg_free=1:7),
            # grid=crossing(max_tokens=c(500,800,1000)),
            # grid=crossing(penalty=10^seq(-7,-1.5,1)),
            metrics=mset,
            control=control_stack_grid())

xg %>% 
  collect_metrics() %>% 
  arrange(mean)

xg %>%
  autoplot() #only for viewing tuning results n check if more tuning is required by looking at straight line

# finalize

xg_test <- xg_wflow %>% 
  finalize_workflow(list( mtry=5,
                          trees=c(700),
                          tree_depth=3,
                          learn_rate=c(0.05))) %>% 
  last_fit(spl)

xg_test %>% collect_metrics() #looking at test values

xg_fit <- xg_wflow %>% 
  finalize_workflow(list(mtry=3,
                         trees=c(700),
                         tree_depth=1,
                         learn_rate=c(0.01))) %>% 
  fit(data)

xg_fit %>% 
  predict(holdout)

attempt2 <- xg_fit %>% 
  predict(holdout) %>% 
  bind_cols(holdout %>% select(Loan_ID)) %>% 
  select(Loan_ID,Loan_Status=.pred_class)

attempt2$Loan_Status <- ifelse(attempt2$Loan_Status==1,'Y','N')

write_csv(attempt2,"C:/Learning/Analytics Vidya/Loan Prediction/attempt2.csv")

# attempt 1 was 0.79 - rank 886 not bad :)
