setwd('/Users/khanh94/Documents/Kaggle/Walmart')

library(data.table)

train <- fread('train.tsv')
test <- fread('test.tsv')

train$tag <- gsub("\\[|\\]", "", train$tag)

train$Seller <- as.numeric(as.factor(train$Seller))
train$'Actual Color' <- as.numeric(as.factor(train$'Actual Color'))
train$'Aspect Ratio' <- as.numeric(as.factor(train$'Aspect Ratio'))
train$'Item Class ID' <- as.numeric(as.factor(train$'Item Class ID'))

train <- train[, ":="(item_id = NULL,
                      Actors = NULL, 
                      #Var1 = NULL,
                      'Artist ID' = NULL,
                      'Literary Genre' = NULL, 
                      ISBN = NULL, 
                      'Product Long Description' = NULL, 
                      'Product Name' = NULL, 
                      'Product Short Description' = NULL, 
                      Publisher = NULL, 
                      'Recommended Location' = NULL,
                      'Recommended Room' = NULL, 
                      'Recommended Use' = NULL, 
                      `Short Description` = NULL, 
                      Synopsis = NULL, 
                      'MPAA Rating' = NULL, 
                      actual_color = NULL, 
                      Color = NULL, 
                      `Genre ID` = NULL
)]

test$Seller <- as.numeric(as.factor(test$Seller))
test$'Actual Color' <- as.numeric(as.factor(test$'Actual Color'))
test$'Aspect Ratio' <- as.numeric(as.factor(test$'Aspect Ratio'))
test$'Item Class ID' <- as.numeric(as.factor(test$'Item Class ID'))

ID <- test$item_id

test <- test[, ":="(item_id = NULL,
                    Actors = NULL, 
                    #Var1 = NULL,
                    'Artist ID' = NULL,
                    'Literary Genre' = NULL, 
                    ISBN = NULL, 
                    'Product Long Description' = NULL, 
                    'Product Name' = NULL, 
                    'Product Short Description' = NULL, 
                    Publisher = NULL, 
                    'Recommended Location' = NULL,
                    'Recommended Room' = NULL, 
                    'Recommended Use' = NULL, 
                    `Short Description` = NULL, 
                    Synopsis = NULL, 
                    'MPAA Rating' = NULL, 
                    actual_color = NULL, 
                    Color = NULL, 
                    `Genre ID` = NULL
)]

train$tag <- as.numeric(train$tag)
train$tag[is.na(train$tag)] = 15


target = as.numeric(as.factor(train$tag)) - 1
outcome <- train$tag
train$tag = NULL

model_xgb_cv <- xgb.cv(data=as.matrix(train), 
                       label=as.matrix(target), 
                       nfold=5, 
                       objective="multi:softmax",
                       num_class = 32,
                       nrounds=600, 
                       eta=0.2, 
                       max_depth=5, 
                       subsample=0.75, 
                       colsample_bytree=0.8, 
                       min_child_weight=1, 
                       eval_metric="mlogloss")


model_xgb <- xgboost(data=as.matrix(train), 
                     label=as.matrix(target), 
                     objective="multi:softmax", 
                     nrounds=600, 
                     num_class = 32,
                     eta=0.2, 
                     max_depth=6, 
                     subsample=0.75, 
                     colsample_bytree=0.8, 
                     min_child_weight=1.5, 
                     booster='gbtree',
                     lambda=2,
                     alpha=1,
                     eval_metric="mlogloss")

preds <- predict(model_xgb, as.matrix(test))

final_sub <- data.frame(item_id = ID, tag = preds)
for(i in 1:length(final_sub$tag)){
  final_sub$tag[i] = as.numeric(levels(as.factor(outcome))[final_sub$tag[i] + 1])}
final_sub$tag[final_sub$tag == 15] = 4537
final_sub$tag <- paste0("[", final_sub$tag,"]")
write.table(final_sub, file='tags.tsv', quote=FALSE, sep='\t', row.names=F)
