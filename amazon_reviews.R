#############################################################################################################
##############################CLEARING THE ENVIRONMENT#######################################################
#############################################################################################################

Sys.setenv(JAVA_HOME = "C:\\Program Files\\Java\\jdk-13.0.2\\")
rm(list = ls())

#############################################################################################################
##############################IMPORTING THE DATA#############################################################
#############################################################################################################

setwd("C:\\Users\\Shekhar Lamba\\Documents\\Datasets")
data <- read.csv("Reviews.csv", stringsAsFactors = F)
data <- na.omit(data)

#############################################################################################################
##############################PREPARING THE RESPONSE VARIABLE FOR TEXT CLASSIFICATION########################
#############################################################################################################

library(ggplot2)
ggplot(data, aes(x = Score)) + geom_bar() + ggtitle("Comparing the frequency of the different scores") +
  ylab("Frequency")
data$Response <- as.factor(ifelse(data$Score > 3, "Positive", "Negative"))
ggplot(data, aes(x = Response)) + geom_bar() + ggtitle("Comparing the frequency of the two responses") +
  xlab("Response") + ylab("Frequency")
amazon <- data[, c(9, 11)]
prop.table(table(data$Response))
amazon_pos <- subset(amazon, Response == "Positive")
amazon_neg <- subset(amazon, Response == "Negative")

#############################################################################################################
##############################CREATING A CORPUS##############################################################
#############################################################################################################

library(NLP)
library(RColorBrewer)
library(tm)
library(ctv)
library(wordcloud)
library(data.table)
reviews <- VCorpus(VectorSource(amazon$Summary))
inspect(reviews[[1]])
reviews_pos <- VCorpus(VectorSource(amazon_pos$Summary))
inspect(reviews_pos[[5]])
reviews_neg <- VCorpus(VectorSource(amazon_neg$Summary))
inspect(reviews_neg[[10]])

#############################################################################################################
##############################CLEANING THE CORPUS############################################################
#############################################################################################################

reviews <- tm_map(reviews, content_transformer(tolower))
reviews <- tm_map(reviews, removeNumbers)
reviews <- tm_map(reviews, removePunctuation)
reviews <- tm_map(reviews, removeWords, stopwords("english"))
reviews <- tm_map(reviews, stemDocument, language = "en")
reviews <- tm_map(reviews, stripWhitespace)
inspect(reviews[[1]])

reviews_pos <- tm_map(reviews_pos, content_transformer(tolower))
reviews_pos <- tm_map(reviews_pos, removeNumbers)
reviews_pos <- tm_map(reviews_pos, removePunctuation)
reviews_pos <- tm_map(reviews_pos, removeWords, stopwords("english"))
reviews_pos <- tm_map(reviews_pos, stemDocument, language = "en")
reviews_pos <- tm_map(reviews_pos, stripWhitespace)
inspect(reviews_pos[[5]])

reviews_neg <- tm_map(reviews_neg, content_transformer(tolower))
reviews_neg <- tm_map(reviews_neg, removeNumbers)
reviews_neg <- tm_map(reviews_neg, removePunctuation)
reviews_neg <- tm_map(reviews_neg, removeWords, stopwords("english"))
reviews_neg <- tm_map(reviews_neg, stemDocument, language = "en")
reviews_neg <- tm_map(reviews_neg, stripWhitespace)
inspect(reviews_neg[[10]])

#############################################################################################################
##############################CREATING DTMS AND WORDCLOUD####################################################
#############################################################################################################

dtm <- DocumentTermMatrix(reviews)
dim(dtm)
dtm_reviews <- removeSparseTerms(dtm, sparse = 0.99965)
dtm_col <- colSums(as.matrix(dtm_reviews))
dtm_features <- data.table(name = attributes(dtm_col)$names, count = dtm_col)

dtm_pos <- DocumentTermMatrix(reviews_pos)
dtm_pos_reviews <- removeSparseTerms(dtm_pos, sparse = 0.99965)
dtm_pos_col <- colSums(as.matrix(dtm_pos_reviews))
dtm_pos_features <- data.table(name = attributes(dtm_pos_col)$names, count = dtm_pos_col)
dtm_pos_features <- dtm_pos_features[order(dtm_pos_features$count, decreasing = T), ]

dtm_neg <- DocumentTermMatrix(reviews_neg)
dtm_neg_reviews <- removeSparseTerms(dtm_neg, sparse = 0.99965)
dtm_neg_col <- colSums(as.matrix(dtm_neg_reviews))
dtm_neg_features <- data.table(name = attributes(dtm_neg_col)$names, count = dtm_neg_col)
dtm_neg_features <- dtm_neg_features[order(dtm_neg_features$count, decreasing = T), ]

ggplot(dtm_pos_features[1:60], aes(x = reorder(name, count), y = count)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_flip() +
  ggtitle("Frequently occuring words in Positive Responses") + xlab("Word") + ylab("Word frequency")
ggplot(dtm_neg_features[1:60], aes(x = reorder(name, count), y = count)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_flip() +
  ggtitle("Frequently occuring words in Negative Responses") + xlab("Word") + ylab("Word frequency")

wordcloud(words = dtm_pos_features$name, freq = dtm_pos_features$count, max.words = 60, scale = c(2,1),
          random.order = F, rot.per = 0.1, colors = brewer.pal(9, "Set1"), random.color = T)
wordcloud(words = dtm_neg_features$name, freq = dtm_neg_features$count, max.words = 60, scale = c(2,1),
          random.order = F, rot.per = 0.35, colors = brewer.pal(8, "Dark2"), random.color = T)

#############################################################################################################
##############################SEPARATING TRAIN AND TEST######################################################
#############################################################################################################

text_reviews <- as.data.table(as.matrix(dtm_reviews))
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
text_reviews <- as.data.frame(apply(text_reviews, MARGIN = 2, convert_counts))
amazon_final <- cbind(data.table(Response = amazon$Response), text_reviews)
set.seed(1234)
index <- sample(1:nrow(amazon_final), size = floor(0.8 * nrow(amazon_final)))
training_data <- amazon_final[index, ]
testing_data <- amazon_final[-index, ]
prop.table(table(training_data$Response))
prop.table(table(testing_data$Response))

#############################################################################################################
##############################FITTING AN H2O NB MODEL########################################################
#############################################################################################################

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "2G")
train <- as.h2o(training_data)
test <- as.h2o(testing_data)
nb_model_1 <- h2o.naiveBayes(x = 2:ncol(train), y = 1, training_frame = train, model_id = "nb_model_1", seed = 1234, nfolds = 10, laplace = 1)
nb_model_1_perf <- h2o.performance(nb_model_1, newdata = test)
h2o.accuracy(nb_model_1_perf)
h2o.confusionMatrix(nb_model_1_perf)
nb_model_1_pred <- h2o.predict(nb_model_1, test)
mean(nb_model_1_pred$predict == test$Response)

#############################################################################################################
##############################USING N-GRAM TOKENIZERS########################################################
#############################################################################################################

library(SnowballC)
library(rJava)
library(RWeka)
NGramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 2:2), paste, collapse = " "), use.names = F)
}
dtm_pos <- DocumentTermMatrix(reviews_pos, control = list(tokenize = NGramTokenizer))
dtm_pos_reviews <- removeSparseTerms(dtm_pos, sparse = 0.99965)
dtm_pos_col <- colSums(as.matrix(dtm_pos_reviews))
dtm_pos_features <- data.table(name = attributes(dtm_pos_col)$name, count = dtm_pos_col)
dtm_pos_features <- dtm_pos_features[order(dtm_pos_features$count, decreasing = T), ]

dtm_neg <- DocumentTermMatrix(reviews_neg, control = list(tokenize = NGramTokenizer))
dtm_neg_reviews <- removeSparseTerms(dtm_neg, sparse = 0.99965)
dtm_neg_col <- colSums(as.matrix(dtm_neg_reviews))
dtm_neg_features <- data.table(name = attributes(dtm_neg_col)$name, count = dtm_neg_col)
dtm_neg_features <- dtm_neg_features[order(dtm_neg_features$count, decreasing = T), ]

ggplot(data = dtm_pos_features[1:60], aes(x = reorder(name, count), y = count)) + geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_flip() +
  ggtitle("Frequently occuring two-word combinations in Positive Responses") + xlab("Two-word combination") + ylab("Frequency")
ggplot(data = dtm_neg_features[1:60], aes(x = reorder(name, count), y = count)) + geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + coord_flip() + 
  ggtitle("Frequently occuring two-word combinations in Negative Responses") + xlab("Two-word combination") + ylab("Frequency")
