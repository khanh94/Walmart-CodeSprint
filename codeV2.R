setwd('/Users/khanh94/Documents/Kaggle/Walmart')

library(data.table)
library(xgboost)
library(tm)
library(tm.plugin.webmining)

train <- fread('train.tsv')
test <- fread('test.tsv')

train$tag <- gsub("\\[|\\]", "", train$tag)

train$Seller <- as.numeric(as.factor(train$Seller))
train$'Actual Color' <- as.numeric(as.factor(train$'Actual Color'))
train$'Aspect Ratio' <- as.numeric(as.factor(train$'Aspect Ratio'))
train$'Item Class ID' <- as.numeric(as.factor(train$'Item Class ID'))

test$Seller <- as.numeric(as.factor(test$Seller))
test$'Actual Color' <- as.numeric(as.factor(test$'Actual Color'))
test$'Aspect Ratio' <- as.numeric(as.factor(test$'Aspect Ratio'))
test$'Item Class ID' <- as.numeric(as.factor(test$'Item Class ID'))

ID <- test$item_id

test$tag <- NA
combi <- rbind(train, test)

corpus <- Corpus(VectorSource(combi$'Product Name'))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, PlainTextDocument)

frequencies <- DocumentTermMatrix(corpus) 
frequencies

sparse <- removeSparseTerms(frequencies, 1 - 10/nrow(frequencies))
dim(sparse)

newsparse <- as.data.frame(as.matrix(sparse))
dim(newsparse)

colnames(newsparse) <- make.names(colnames(newsparse))

train$tag <- as.numeric(train$tag)
train$tag[is.na(train$tag)] = 15

target = as.numeric(as.factor(train$tag)) - 1
outcome <- train$tag
#train$tag = NULL

mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]

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

mytrain <- cbind(mytrain, train)
mytest <- cbind(mytest, test)

target = as.numeric(as.factor(mytrain$tag)) - 1
outcome <- mytrain$tag
mytrain$tag = NULL
mytest$tag = NULL

model_xgb_cv <- xgb.cv(data=as.matrix(mytrain), 
                       label=as.matrix(target), 
                       nfold=5, 
                       objective="multi:softmax",
                       num_class = 32,
                       nrounds=1, 
                       eta=0.2, 
                       max_depth=5, 
                       subsample=0.75, 
                       colsample_bytree=0.8, 
                       min_child_weight=1, 
                       eval_metric="merror")


model_xgb <- xgboost(data=as.matrix(mytrain), 
                     label=as.matrix(target), 
                     objective="multi:softmax", 
                     nrounds=20, 
                     num_class = 32,
                     eta=0.2, 
                     max_depth=6, 
                     subsample=0.75, 
                     colsample_bytree=0.8, 
                     min_child_weight=1.5, 
                     booster='gbtree',
                     lambda=2,
                     alpha=1,
                     eval_metric="merror")

preds <- predict(model_xgb, as.matrix(mytest))

final_sub <- data.frame(item_id = ID, tag = preds)
for(i in 1:length(final_sub$tag)){
   final_sub$tag[i] = as.numeric(levels(as.factor(outcome))[final_sub$tag[i] + 1])}

final_sub$tag[final_sub$tag == 15] = 4537
final_sub$tag <- paste0("[", final_sub$tag,"]")
write.table(final_sub, file='tags.tsv', quote=FALSE, sep='\t', row.names=F)
