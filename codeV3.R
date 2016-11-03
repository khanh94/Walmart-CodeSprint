setwd('/Users/khanh94/Documents/Kaggle/Walmart')

library(data.table)
library(xgboost)
library(tm)
library(tm.plugin.webmining)
library(qdap)
library(RWeka)
library(dplyr)

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

corpus = Corpus(DataframeSource(combi %>% select(13, 14, 15)))
corpus <- tm_map(corpus, tolower)
#corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, PlainTextDocument)


frequencies <- DocumentTermMatrix(corpus) #control = list(tokenize = BigramTokenizer)) 

sparse <- removeSparseTerms(frequencies, 1 - 30/nrow(frequencies))
dim(sparse)

newsparse <- as.data.frame(as.matrix(sparse))
#newsparse2 <- as.data.frame(as.matrix(sparse2))
#newsparse <- cbind(newsparse, newsparse2)
dim(newsparse)

colnames(newsparse) <- make.names(colnames(newsparse))


for(i in 1:length(train$tag)){
  if(length(strsplit(train$tag, ",")[i][[1]]) == 2){
    train$tag[i] = 2
  }
}

for(i in 1:length(train$tag)){
  if(length(strsplit(train$tag, ",")[i][[1]]) == 3){
    train$tag[i] = 3
  }
}


for(i in 1:length(train$tag)){
  if(length(strsplit(train$tag, ",")[i][[1]]) == 4){
    train$tag[i] = 4
  }
}

for(i in 1:length(train$tag)){
  if(length(strsplit(train$tag, ",")[i][[1]]) == 5){
    train$tag[i] = 5
  }
}

for(i in 1:length(train$tag)){
  if(length(strsplit(train$tag, ",")[i][[1]]) == 6){
    train$tag[i] = 6
  }
}

train$tag <- as.numeric(train$tag)
#train$tag[is.na(train$tag)] = 15

target = as.numeric(as.factor(train$tag)) - 1
outcome <- train$tag

mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]

rm(corpus)
rm(corpus2)
rm(frequencies)
rm(frequencies2)
rm(sparse) 
rm(newsparse)
rm(combi)
gc()

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
                       num_class = 37,
                       nrounds=1, 
                       eta=0.5, 
                       max_depth=8, 
                       subsample=0.75, 
                       colsample_bytree=0.8, 
                       min_child_weight=1, 
                       eval_metric="merror")


model_xgb <- xgboost(data=as.matrix(mytrain), 
                     label=as.matrix(target), 
                     objective="multi:softmax", 
                     nrounds=60, 
                     num_class = 37,
                     eta=0.5, 
                     max_depth=8, 
                     subsample=0.75, 
                     colsample_bytree=0.8, 
                     min_child_weight=1.5, 
                     booster='gbtree',
                     lambda=2,
                     alpha=1,
                     eval_metric="merror")

preds <- predict(model_xgb, as.matrix(mytest))

rm(mytrain)
gc()
rm(mytest)
gc()
rm(train)
gc()
rm(test)
gc()

final_sub <- data.frame(item_id = ID, tag = preds)
for(i in 1:length(final_sub$tag)){
  final_sub$tag[i] = as.numeric(levels(as.factor(outcome))[final_sub$tag[i] + 1])}

final_sub$tag[final_sub$tag == 2] = paste0('3304195, ', '1229821')
final_sub$tag[final_sub$tag == 3] = paste0('522484, ', '106546, ', '127175')  
final_sub$tag[final_sub$tag == 4] = paste0('95987, ', '522484, ', '106546, ', '127175')
final_sub$tag[final_sub$tag == 5] = paste0('1229817, ', '1229820, ', '1180168, ', '1229821, ', '447913')  
final_sub$tag[final_sub$tag == 6] = paste0('3304195, ', '1229818, ', '447913, ', '1229817, ', '1180168, ', '1229821')

final_sub$tag <- paste0("[", final_sub$tag,"]")
write.table(final_sub, file='tags.tsv', quote=FALSE, sep='\t', row.names=F)