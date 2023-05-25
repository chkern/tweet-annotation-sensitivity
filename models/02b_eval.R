##################
# ML Label Quality
# Evaluate models -- sampled
# R 3.6.3
##################

## Setup

library(tidyverse)
library(mlr3)
library(mlr3measures)
library(GGally)

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/preds_sampled")

lstm_testA <- read_csv(file = "lstm_testA_sampled.csv")
lstm_testB <- read_csv(file = "lstm_testB_sampled.csv")
lstm_testC <- read_csv(file = "lstm_testC_sampled.csv")
lstm_testD <- read_csv(file = "lstm_testD_sampled.csv")
lstm_testE <- read_csv(file = "lstm_testE_sampled.csv")

bert_testA <- read_csv(file = "bert_testA_sampled.csv")
bert_testB <- read_csv(file = "bert_testB_sampled.csv")
bert_testC <- read_csv(file = "bert_testC_sampled.csv")
bert_testD <- read_csv(file = "bert_testD_sampled.csv")
bert_testE <- read_csv(file = "bert_testE_sampled.csv")

## 01 Prepare Data

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

lstm_testA <- lstm_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

lstm_testA_mode <- lstm_testA %>%
  group_by(tweet.id) %>%
  summarise_if(is.factor, Mode)
  
lstm_testB <- lstm_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

lstm_testB_mode <- lstm_testB %>%
  group_by(tweet.id) %>%
  summarise_if(is.factor, Mode)

lstm_testC <- lstm_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

lstm_testC_mode <- lstm_testC %>%
  group_by(tweet.id) %>%
  summarise_if(is.factor, Mode)

lstm_testD <- lstm_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

lstm_testD_mode <- lstm_testD %>%
  group_by(tweet.id) %>%
  summarise_if(is.factor, Mode)

lstm_testE <- lstm_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

lstm_testE_mode <- lstm_testE %>%
  group_by(tweet.id) %>%
  summarise_if(is.factor, Mode)

lstm_test <- lstm_testA %>% 
  bind_rows(lstm_testB, lstm_testC, lstm_testD, lstm_testE) %>%
  drop_na()

bert_testA <- bert_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

bert_testB <- bert_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

bert_testC <- bert_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

bert_testD <- bert_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

bert_testE <- bert_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         hate_speech_preds_A = factor(ifelse(hate.speech_preds_A == 1, "yes", "no")),
         hate_speech_preds_B = factor(ifelse(hate.speech_preds_B == 1, "yes", "no")),
         hate_speech_preds_C = factor(ifelse(hate.speech_preds_C == 1, "yes", "no")),
         hate_speech_preds_D = factor(ifelse(hate.speech_preds_D == 1, "yes", "no")),
         hate_speech_preds_E = factor(ifelse(hate.speech_preds_E == 1, "yes", "no")),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)),
         offensive_language_preds_A = factor(ifelse(offensive.language_preds_A == 1, "yes", "no")),
         offensive_language_preds_B = factor(ifelse(offensive.language_preds_B == 1, "yes", "no")),
         offensive_language_preds_C = factor(ifelse(offensive.language_preds_C == 1, "yes", "no")),
         offensive_language_preds_D = factor(ifelse(offensive.language_preds_D == 1, "yes", "no")),
         offensive_language_preds_E = factor(ifelse(offensive.language_preds_E == 1, "yes", "no")))

bert_test <- bert_testA %>% 
  bind_rows(bert_testB, bert_testC, bert_testD, bert_testE) %>%
  drop_na()

## 02 Compare Classification Performance
### Hate Speech

lstm_hate <- data.frame(bacc = rep(NA, 25),
                        auc = rep(NA, 25),
                        test = rep(c("A", "B", "C", "D", "E"), each = 5),
                        train = rep(c("A", "B", "C", "D", "E"), 5))

lstm_hate$bacc[1] <- bacc(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_A)
lstm_hate$bacc[2] <- bacc(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_B) 
lstm_hate$bacc[3] <- bacc(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_C)
lstm_hate$bacc[4] <- bacc(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_D)
lstm_hate$bacc[5] <- bacc(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_E)

lstm_hate$bacc[6] <- bacc(lstm_testB$hate_speech, lstm_testB$hate_speech_preds_A)
lstm_hate$bacc[7] <- bacc(lstm_testB$hate_speech, lstm_testB$hate_speech_preds_B) 
lstm_hate$bacc[8] <- bacc(lstm_testB$hate_speech, lstm_testB$hate_speech_preds_C)
lstm_hate$bacc[9] <- bacc(lstm_testB$hate_speech, lstm_testB$hate_speech_preds_D)
lstm_hate$bacc[10] <- bacc(lstm_testB$hate_speech, lstm_testB$hate_speech_preds_E)

lstm_hate$bacc[11] <- bacc(lstm_testC$hate_speech, lstm_testC$hate_speech_preds_A)
lstm_hate$bacc[12] <- bacc(lstm_testC$hate_speech, lstm_testC$hate_speech_preds_B) 
lstm_hate$bacc[13] <- bacc(lstm_testC$hate_speech, lstm_testC$hate_speech_preds_C)
lstm_hate$bacc[14] <- bacc(lstm_testC$hate_speech, lstm_testC$hate_speech_preds_D)
lstm_hate$bacc[15] <- bacc(lstm_testC$hate_speech, lstm_testC$hate_speech_preds_E)

lstm_hate$bacc[16] <- bacc(lstm_testD$hate_speech, lstm_testD$hate_speech_preds_A)
lstm_hate$bacc[17] <- bacc(lstm_testD$hate_speech, lstm_testD$hate_speech_preds_B) 
lstm_hate$bacc[18] <- bacc(lstm_testD$hate_speech, lstm_testD$hate_speech_preds_C)
lstm_hate$bacc[19] <- bacc(lstm_testD$hate_speech, lstm_testD$hate_speech_preds_D)
lstm_hate$bacc[20] <- bacc(lstm_testD$hate_speech, lstm_testD$hate_speech_preds_E)

lstm_hate$bacc[21] <- bacc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate_speech_preds_A[!is.na(lstm_testE$hate_speech)])
lstm_hate$bacc[22] <- bacc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate_speech_preds_B[!is.na(lstm_testE$hate_speech)]) 
lstm_hate$bacc[23] <- bacc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate_speech_preds_C[!is.na(lstm_testE$hate_speech)])
lstm_hate$bacc[24] <- bacc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate_speech_preds_D[!is.na(lstm_testE$hate_speech)])
lstm_hate$bacc[25] <- bacc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate_speech_preds_E[!is.na(lstm_testE$hate_speech)])

lstm_hate$auc[1] <- auc(lstm_testA$hate_speech, lstm_testA$hate.speech_preds_A_scores, positive = "yes")
lstm_hate$auc[2] <- auc(lstm_testA$hate_speech, lstm_testA$hate.speech_preds_B_scores, positive = "yes")
lstm_hate$auc[3] <- auc(lstm_testA$hate_speech, lstm_testA$hate.speech_preds_C_scores, positive = "yes")
lstm_hate$auc[4] <- auc(lstm_testA$hate_speech, lstm_testA$hate.speech_preds_D_scores, positive = "yes")
lstm_hate$auc[5] <- auc(lstm_testA$hate_speech, lstm_testA$hate.speech_preds_E_scores, positive = "yes")

lstm_hate$auc[6] <- auc(lstm_testB$hate_speech, lstm_testB$hate.speech_preds_A_scores, positive = "yes")
lstm_hate$auc[7] <- auc(lstm_testB$hate_speech, lstm_testB$hate.speech_preds_B_scores, positive = "yes")
lstm_hate$auc[8] <- auc(lstm_testB$hate_speech, lstm_testB$hate.speech_preds_C_scores, positive = "yes")
lstm_hate$auc[9] <- auc(lstm_testB$hate_speech, lstm_testB$hate.speech_preds_D_scores, positive = "yes")
lstm_hate$auc[10] <- auc(lstm_testB$hate_speech, lstm_testB$hate.speech_preds_E_scores, positive = "yes")

lstm_hate$auc[11] <- auc(lstm_testC$hate_speech, lstm_testC$hate.speech_preds_A_scores, positive = "yes")
lstm_hate$auc[12] <- auc(lstm_testC$hate_speech, lstm_testC$hate.speech_preds_B_scores, positive = "yes")
lstm_hate$auc[13] <- auc(lstm_testC$hate_speech, lstm_testC$hate.speech_preds_C_scores, positive = "yes")
lstm_hate$auc[14] <- auc(lstm_testC$hate_speech, lstm_testC$hate.speech_preds_D_scores, positive = "yes")
lstm_hate$auc[15] <- auc(lstm_testC$hate_speech, lstm_testC$hate.speech_preds_E_scores, positive = "yes")

lstm_hate$auc[16] <- auc(lstm_testD$hate_speech, lstm_testD$hate.speech_preds_A_scores, positive = "yes")
lstm_hate$auc[17] <- auc(lstm_testD$hate_speech, lstm_testD$hate.speech_preds_B_scores, positive = "yes")
lstm_hate$auc[18] <- auc(lstm_testD$hate_speech, lstm_testD$hate.speech_preds_C_scores, positive = "yes")
lstm_hate$auc[19] <- auc(lstm_testD$hate_speech, lstm_testD$hate.speech_preds_D_scores, positive = "yes")
lstm_hate$auc[20] <- auc(lstm_testD$hate_speech, lstm_testD$hate.speech_preds_E_scores, positive = "yes")

lstm_hate$auc[21] <- auc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate.speech_preds_A_scores[!is.na(lstm_testE$hate_speech)], positive = "yes")
lstm_hate$auc[22] <- auc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate.speech_preds_B_scores[!is.na(lstm_testE$hate_speech)], positive = "yes")
lstm_hate$auc[23] <- auc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate.speech_preds_C_scores[!is.na(lstm_testE$hate_speech)], positive = "yes")
lstm_hate$auc[24] <- auc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate.speech_preds_D_scores[!is.na(lstm_testE$hate_speech)], positive = "yes")
lstm_hate$auc[25] <- auc(lstm_testE$hate_speech[!is.na(lstm_testE$hate_speech)], lstm_testE$hate.speech_preds_E_scores[!is.na(lstm_testE$hate_speech)], positive = "yes")

lstm_f1_A_A <- fbeta(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_A, positive = "yes")

bert_hate <- data.frame(bacc = rep(NA, 25),
                        auc = rep(NA, 25),
                        test = rep(c("A", "B", "C", "D", "E"), each = 5),
                        train = rep(c("A", "B", "C", "D", "E"), 5))

bert_hate$bacc[1] <- bacc(bert_testA$hate_speech, bert_testA$hate_speech_preds_A)
bert_hate$bacc[2] <- bacc(bert_testA$hate_speech, bert_testA$hate_speech_preds_B) 
bert_hate$bacc[3] <- bacc(bert_testA$hate_speech, bert_testA$hate_speech_preds_C)
bert_hate$bacc[4] <- bacc(bert_testA$hate_speech, bert_testA$hate_speech_preds_D)
bert_hate$bacc[5] <- bacc(bert_testA$hate_speech, bert_testA$hate_speech_preds_E)

bert_hate$bacc[6] <- bacc(bert_testB$hate_speech, bert_testB$hate_speech_preds_A)
bert_hate$bacc[7] <- bacc(bert_testB$hate_speech, bert_testB$hate_speech_preds_B) 
bert_hate$bacc[8] <- bacc(bert_testB$hate_speech, bert_testB$hate_speech_preds_C)
bert_hate$bacc[9] <- bacc(bert_testB$hate_speech, bert_testB$hate_speech_preds_D)
bert_hate$bacc[10] <- bacc(bert_testB$hate_speech, bert_testB$hate_speech_preds_E)

bert_hate$bacc[11] <- bacc(bert_testC$hate_speech, bert_testC$hate_speech_preds_A)
bert_hate$bacc[12] <- bacc(bert_testC$hate_speech, bert_testC$hate_speech_preds_B) 
bert_hate$bacc[13] <- bacc(bert_testC$hate_speech, bert_testC$hate_speech_preds_C)
bert_hate$bacc[14] <- bacc(bert_testC$hate_speech, bert_testC$hate_speech_preds_D)
bert_hate$bacc[15] <- bacc(bert_testC$hate_speech, bert_testC$hate_speech_preds_E)

bert_hate$bacc[16] <- bacc(bert_testD$hate_speech, bert_testD$hate_speech_preds_A)
bert_hate$bacc[17] <- bacc(bert_testD$hate_speech, bert_testD$hate_speech_preds_B) 
bert_hate$bacc[18] <- bacc(bert_testD$hate_speech, bert_testD$hate_speech_preds_C)
bert_hate$bacc[19] <- bacc(bert_testD$hate_speech, bert_testD$hate_speech_preds_D)
bert_hate$bacc[20] <- bacc(bert_testD$hate_speech, bert_testD$hate_speech_preds_E)

bert_hate$bacc[21] <- bacc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate_speech_preds_A[!is.na(bert_testE$hate_speech)])
bert_hate$bacc[22] <- bacc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate_speech_preds_B[!is.na(bert_testE$hate_speech)]) 
bert_hate$bacc[23] <- bacc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate_speech_preds_C[!is.na(bert_testE$hate_speech)])
bert_hate$bacc[24] <- bacc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate_speech_preds_D[!is.na(bert_testE$hate_speech)])
bert_hate$bacc[25] <- bacc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate_speech_preds_E[!is.na(bert_testE$hate_speech)])

bert_hate$auc[1] <- auc(bert_testA$hate_speech, bert_testA$hate.speech_preds_A_scores, positive = "yes")
bert_hate$auc[2] <- auc(bert_testA$hate_speech, bert_testA$hate.speech_preds_B_scores, positive = "yes")
bert_hate$auc[3] <- auc(bert_testA$hate_speech, bert_testA$hate.speech_preds_C_scores, positive = "yes")
bert_hate$auc[4] <- auc(bert_testA$hate_speech, bert_testA$hate.speech_preds_D_scores, positive = "yes")
bert_hate$auc[5] <- auc(bert_testA$hate_speech, bert_testA$hate.speech_preds_E_scores, positive = "yes")

bert_hate$auc[6] <- auc(bert_testB$hate_speech, bert_testB$hate.speech_preds_A_scores, positive = "yes")
bert_hate$auc[7] <- auc(bert_testB$hate_speech, bert_testB$hate.speech_preds_B_scores, positive = "yes")
bert_hate$auc[8] <- auc(bert_testB$hate_speech, bert_testB$hate.speech_preds_C_scores, positive = "yes")
bert_hate$auc[9] <- auc(bert_testB$hate_speech, bert_testB$hate.speech_preds_D_scores, positive = "yes")
bert_hate$auc[10] <- auc(bert_testB$hate_speech, bert_testB$hate.speech_preds_E_scores, positive = "yes")

bert_hate$auc[11] <- auc(bert_testC$hate_speech, bert_testC$hate.speech_preds_A_scores, positive = "yes")
bert_hate$auc[12] <- auc(bert_testC$hate_speech, bert_testC$hate.speech_preds_B_scores, positive = "yes")
bert_hate$auc[13] <- auc(bert_testC$hate_speech, bert_testC$hate.speech_preds_C_scores, positive = "yes")
bert_hate$auc[14] <- auc(bert_testC$hate_speech, bert_testC$hate.speech_preds_D_scores, positive = "yes")
bert_hate$auc[15] <- auc(bert_testC$hate_speech, bert_testC$hate.speech_preds_E_scores, positive = "yes")

bert_hate$auc[16] <- auc(bert_testD$hate_speech, bert_testD$hate.speech_preds_A_scores, positive = "yes")
bert_hate$auc[17] <- auc(bert_testD$hate_speech, bert_testD$hate.speech_preds_B_scores, positive = "yes")
bert_hate$auc[18] <- auc(bert_testD$hate_speech, bert_testD$hate.speech_preds_C_scores, positive = "yes")
bert_hate$auc[19] <- auc(bert_testD$hate_speech, bert_testD$hate.speech_preds_D_scores, positive = "yes")
bert_hate$auc[20] <- auc(bert_testD$hate_speech, bert_testD$hate.speech_preds_E_scores, positive = "yes")

bert_hate$auc[21] <- auc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate.speech_preds_A_scores[!is.na(bert_testE$hate_speech)], positive = "yes")
bert_hate$auc[22] <- auc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate.speech_preds_B_scores[!is.na(bert_testE$hate_speech)], positive = "yes")
bert_hate$auc[23] <- auc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate.speech_preds_C_scores[!is.na(bert_testE$hate_speech)], positive = "yes")
bert_hate$auc[24] <- auc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate.speech_preds_D_scores[!is.na(bert_testE$hate_speech)], positive = "yes")
bert_hate$auc[25] <- auc(bert_testE$hate_speech[!is.na(bert_testE$hate_speech)], bert_testE$hate.speech_preds_E_scores[!is.na(bert_testE$hate_speech)], positive = "yes")

bert_f1_A_A <- fbeta(bert_testA$hate_speech, bert_testA$hate_speech_preds_A, positive = "yes")

### Offensive Language

lstm_offensive <- data.frame(bacc = rep(NA, 25),
                        auc = rep(NA, 25),
                        test = rep(c("A", "B", "C", "D", "E"), each = 5),
                        train = rep(c("A", "B", "C", "D", "E"), 5))

lstm_offensive$bacc[1] <- bacc(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_A)
lstm_offensive$bacc[2] <- bacc(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_B) 
lstm_offensive$bacc[3] <- bacc(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_C)
lstm_offensive$bacc[4] <- bacc(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_D)
lstm_offensive$bacc[5] <- bacc(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_E)

lstm_offensive$bacc[6] <- bacc(lstm_testB$offensive_language, lstm_testB$offensive_language_preds_A)
lstm_offensive$bacc[7] <- bacc(lstm_testB$offensive_language, lstm_testB$offensive_language_preds_B) 
lstm_offensive$bacc[8] <- bacc(lstm_testB$offensive_language, lstm_testB$offensive_language_preds_C)
lstm_offensive$bacc[9] <- bacc(lstm_testB$offensive_language, lstm_testB$offensive_language_preds_D)
lstm_offensive$bacc[10] <- bacc(lstm_testB$offensive_language, lstm_testB$offensive_language_preds_E)

lstm_offensive$bacc[11] <- bacc(lstm_testC$offensive_language, lstm_testC$offensive_language_preds_A)
lstm_offensive$bacc[12] <- bacc(lstm_testC$offensive_language, lstm_testC$offensive_language_preds_B) 
lstm_offensive$bacc[13] <- bacc(lstm_testC$offensive_language, lstm_testC$offensive_language_preds_C)
lstm_offensive$bacc[14] <- bacc(lstm_testC$offensive_language, lstm_testC$offensive_language_preds_D)
lstm_offensive$bacc[15] <- bacc(lstm_testC$offensive_language, lstm_testC$offensive_language_preds_E)

lstm_offensive$bacc[16] <- bacc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive_language_preds_A[!is.na(lstm_testD$offensive_language)])
lstm_offensive$bacc[17] <- bacc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive_language_preds_B[!is.na(lstm_testD$offensive_language)]) 
lstm_offensive$bacc[18] <- bacc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive_language_preds_C[!is.na(lstm_testD$offensive_language)])
lstm_offensive$bacc[19] <- bacc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive_language_preds_D[!is.na(lstm_testD$offensive_language)])
lstm_offensive$bacc[20] <- bacc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive_language_preds_E[!is.na(lstm_testD$offensive_language)])

lstm_offensive$bacc[21] <- bacc(lstm_testE$offensive_language, lstm_testE$offensive_language_preds_A)
lstm_offensive$bacc[22] <- bacc(lstm_testE$offensive_language, lstm_testE$offensive_language_preds_B) 
lstm_offensive$bacc[23] <- bacc(lstm_testE$offensive_language, lstm_testE$offensive_language_preds_C)
lstm_offensive$bacc[24] <- bacc(lstm_testE$offensive_language, lstm_testE$offensive_language_preds_D)
lstm_offensive$bacc[25] <- bacc(lstm_testE$offensive_language, lstm_testE$offensive_language_preds_E)

lstm_offensive$auc[1] <- auc(lstm_testA$offensive_language, lstm_testA$offensive.language_preds_A_scores, positive = "yes")
lstm_offensive$auc[2] <- auc(lstm_testA$offensive_language, lstm_testA$offensive.language_preds_B_scores, positive = "yes")
lstm_offensive$auc[3] <- auc(lstm_testA$offensive_language, lstm_testA$offensive.language_preds_C_scores, positive = "yes")
lstm_offensive$auc[4] <- auc(lstm_testA$offensive_language, lstm_testA$offensive.language_preds_D_scores, positive = "yes")
lstm_offensive$auc[5] <- auc(lstm_testA$offensive_language, lstm_testA$offensive.language_preds_E_scores, positive = "yes")

lstm_offensive$auc[6] <- auc(lstm_testB$offensive_language, lstm_testB$offensive.language_preds_A_scores, positive = "yes")
lstm_offensive$auc[7] <- auc(lstm_testB$offensive_language, lstm_testB$offensive.language_preds_B_scores, positive = "yes")
lstm_offensive$auc[8] <- auc(lstm_testB$offensive_language, lstm_testB$offensive.language_preds_C_scores, positive = "yes")
lstm_offensive$auc[9] <- auc(lstm_testB$offensive_language, lstm_testB$offensive.language_preds_D_scores, positive = "yes")
lstm_offensive$auc[10] <- auc(lstm_testB$offensive_language, lstm_testB$offensive.language_preds_E_scores, positive = "yes")

lstm_offensive$auc[11] <- auc(lstm_testC$offensive_language, lstm_testC$offensive.language_preds_A_scores, positive = "yes")
lstm_offensive$auc[12] <- auc(lstm_testC$offensive_language, lstm_testC$offensive.language_preds_B_scores, positive = "yes")
lstm_offensive$auc[13] <- auc(lstm_testC$offensive_language, lstm_testC$offensive.language_preds_C_scores, positive = "yes")
lstm_offensive$auc[14] <- auc(lstm_testC$offensive_language, lstm_testC$offensive.language_preds_D_scores, positive = "yes")
lstm_offensive$auc[15] <- auc(lstm_testC$offensive_language, lstm_testC$offensive.language_preds_E_scores, positive = "yes")

lstm_offensive$auc[16] <- auc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive.language_preds_A_scores[!is.na(lstm_testD$offensive_language)], positive = "yes")
lstm_offensive$auc[17] <- auc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive.language_preds_B_scores[!is.na(lstm_testD$offensive_language)], positive = "yes")
lstm_offensive$auc[18] <- auc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive.language_preds_C_scores[!is.na(lstm_testD$offensive_language)], positive = "yes")
lstm_offensive$auc[19] <- auc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive.language_preds_D_scores[!is.na(lstm_testD$offensive_language)], positive = "yes")
lstm_offensive$auc[20] <- auc(lstm_testD$offensive_language[!is.na(lstm_testD$offensive_language)], lstm_testD$offensive.language_preds_E_scores[!is.na(lstm_testD$offensive_language)], positive = "yes")

lstm_offensive$auc[21] <- auc(lstm_testE$offensive_language, lstm_testE$offensive.language_preds_A_scores, positive = "yes")
lstm_offensive$auc[22] <- auc(lstm_testE$offensive_language, lstm_testE$offensive.language_preds_B_scores, positive = "yes")
lstm_offensive$auc[23] <- auc(lstm_testE$offensive_language, lstm_testE$offensive.language_preds_C_scores, positive = "yes")
lstm_offensive$auc[24] <- auc(lstm_testE$offensive_language, lstm_testE$offensive.language_preds_D_scores, positive = "yes")
lstm_offensive$auc[25] <- auc(lstm_testE$offensive_language, lstm_testE$offensive.language_preds_E_scores, positive = "yes")

lstm_f1_A_A <- fbeta(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_A, positive = "yes")

bert_offensive <- data.frame(bacc = rep(NA, 25),
                             auc = rep(NA, 25),
                             test = rep(c("A", "B", "C", "D", "E"), each = 5),
                             train = rep(c("A", "B", "C", "D", "E"), 5))

bert_offensive$bacc[1] <- bacc(bert_testA$offensive_language, bert_testA$offensive_language_preds_A)
bert_offensive$bacc[2] <- bacc(bert_testA$offensive_language, bert_testA$offensive_language_preds_B) 
bert_offensive$bacc[3] <- bacc(bert_testA$offensive_language, bert_testA$offensive_language_preds_C)
bert_offensive$bacc[4] <- bacc(bert_testA$offensive_language, bert_testA$offensive_language_preds_D)
bert_offensive$bacc[5] <- bacc(bert_testA$offensive_language, bert_testA$offensive_language_preds_E)

bert_offensive$bacc[6] <- bacc(bert_testB$offensive_language, bert_testB$offensive_language_preds_A)
bert_offensive$bacc[7] <- bacc(bert_testB$offensive_language, bert_testB$offensive_language_preds_B) 
bert_offensive$bacc[8] <- bacc(bert_testB$offensive_language, bert_testB$offensive_language_preds_C)
bert_offensive$bacc[9] <- bacc(bert_testB$offensive_language, bert_testB$offensive_language_preds_D)
bert_offensive$bacc[10] <- bacc(bert_testB$offensive_language, bert_testB$offensive_language_preds_E)

bert_offensive$bacc[11] <- bacc(bert_testC$offensive_language, bert_testC$offensive_language_preds_A)
bert_offensive$bacc[12] <- bacc(bert_testC$offensive_language, bert_testC$offensive_language_preds_B) 
bert_offensive$bacc[13] <- bacc(bert_testC$offensive_language, bert_testC$offensive_language_preds_C)
bert_offensive$bacc[14] <- bacc(bert_testC$offensive_language, bert_testC$offensive_language_preds_D)
bert_offensive$bacc[15] <- bacc(bert_testC$offensive_language, bert_testC$offensive_language_preds_E)

bert_offensive$bacc[16] <- bacc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive_language_preds_A[!is.na(bert_testD$offensive_language)])
bert_offensive$bacc[17] <- bacc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive_language_preds_B[!is.na(bert_testD$offensive_language)]) 
bert_offensive$bacc[18] <- bacc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive_language_preds_C[!is.na(bert_testD$offensive_language)])
bert_offensive$bacc[19] <- bacc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive_language_preds_D[!is.na(bert_testD$offensive_language)])
bert_offensive$bacc[20] <- bacc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive_language_preds_E[!is.na(bert_testD$offensive_language)])

bert_offensive$bacc[21] <- bacc(bert_testE$offensive_language, bert_testE$offensive_language_preds_A)
bert_offensive$bacc[22] <- bacc(bert_testE$offensive_language, bert_testE$offensive_language_preds_B) 
bert_offensive$bacc[23] <- bacc(bert_testE$offensive_language, bert_testE$offensive_language_preds_C)
bert_offensive$bacc[24] <- bacc(bert_testE$offensive_language, bert_testE$offensive_language_preds_D)
bert_offensive$bacc[25] <- bacc(bert_testE$offensive_language, bert_testE$offensive_language_preds_E)

bert_offensive$auc[1] <- auc(bert_testA$offensive_language, bert_testA$offensive.language_preds_A_scores, positive = "yes")
bert_offensive$auc[2] <- auc(bert_testA$offensive_language, bert_testA$offensive.language_preds_B_scores, positive = "yes")
bert_offensive$auc[3] <- auc(bert_testA$offensive_language, bert_testA$offensive.language_preds_C_scores, positive = "yes")
bert_offensive$auc[4] <- auc(bert_testA$offensive_language, bert_testA$offensive.language_preds_D_scores, positive = "yes")
bert_offensive$auc[5] <- auc(bert_testA$offensive_language, bert_testA$offensive.language_preds_E_scores, positive = "yes")

bert_offensive$auc[6] <- auc(bert_testB$offensive_language, bert_testB$offensive.language_preds_A_scores, positive = "yes")
bert_offensive$auc[7] <- auc(bert_testB$offensive_language, bert_testB$offensive.language_preds_B_scores, positive = "yes")
bert_offensive$auc[8] <- auc(bert_testB$offensive_language, bert_testB$offensive.language_preds_C_scores, positive = "yes")
bert_offensive$auc[9] <- auc(bert_testB$offensive_language, bert_testB$offensive.language_preds_D_scores, positive = "yes")
bert_offensive$auc[10] <- auc(bert_testB$offensive_language, bert_testB$offensive.language_preds_E_scores, positive = "yes")

bert_offensive$auc[11] <- auc(bert_testC$offensive_language, bert_testC$offensive.language_preds_A_scores, positive = "yes")
bert_offensive$auc[12] <- auc(bert_testC$offensive_language, bert_testC$offensive.language_preds_B_scores, positive = "yes")
bert_offensive$auc[13] <- auc(bert_testC$offensive_language, bert_testC$offensive.language_preds_C_scores, positive = "yes")
bert_offensive$auc[14] <- auc(bert_testC$offensive_language, bert_testC$offensive.language_preds_D_scores, positive = "yes")
bert_offensive$auc[15] <- auc(bert_testC$offensive_language, bert_testC$offensive.language_preds_E_scores, positive = "yes")

bert_offensive$auc[16] <- auc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive.language_preds_A_scores[!is.na(bert_testD$offensive_language)], positive = "yes")
bert_offensive$auc[17] <- auc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive.language_preds_B_scores[!is.na(bert_testD$offensive_language)], positive = "yes")
bert_offensive$auc[18] <- auc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive.language_preds_C_scores[!is.na(bert_testD$offensive_language)], positive = "yes")
bert_offensive$auc[19] <- auc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive.language_preds_D_scores[!is.na(bert_testD$offensive_language)], positive = "yes")
bert_offensive$auc[20] <- auc(bert_testD$offensive_language[!is.na(bert_testD$offensive_language)], bert_testD$offensive.language_preds_E_scores[!is.na(bert_testD$offensive_language)], positive = "yes")

bert_offensive$auc[21] <- auc(bert_testE$offensive_language, bert_testE$offensive.language_preds_A_scores, positive = "yes")
bert_offensive$auc[22] <- auc(bert_testE$offensive_language, bert_testE$offensive.language_preds_B_scores, positive = "yes")
bert_offensive$auc[23] <- auc(bert_testE$offensive_language, bert_testE$offensive.language_preds_C_scores, positive = "yes")
bert_offensive$auc[24] <- auc(bert_testE$offensive_language, bert_testE$offensive.language_preds_D_scores, positive = "yes")
bert_offensive$auc[25] <- auc(bert_testE$offensive_language, bert_testE$offensive.language_preds_E_scores, positive = "yes")

bert_f1_A_A <- fbeta(bert_testA$offensive_language, bert_testA$offensive_language_preds_A, positive = "yes")

## Mode

lstm_ol_mode <- data.frame(bacc = rep(NA, 25),
                           test = rep(c("A", "B", "C", "D", "E"), each = 5),
                           train = rep(c("A", "B", "C", "D", "E"), 5))

lstm_ol_mode$bacc[1] <- bacc(lstm_testA_mode$offensive_language, lstm_testA_mode$offensive_language_preds_A)
lstm_ol_mode$bacc[2] <- bacc(lstm_testA_mode$offensive_language, lstm_testA_mode$offensive_language_preds_B) 
lstm_ol_mode$bacc[3] <- bacc(lstm_testA_mode$offensive_language, lstm_testA_mode$offensive_language_preds_C)
lstm_ol_mode$bacc[4] <- bacc(lstm_testA_mode$offensive_language, lstm_testA_mode$offensive_language_preds_D)
lstm_ol_mode$bacc[5] <- bacc(lstm_testA_mode$offensive_language, lstm_testA_mode$offensive_language_preds_E)

lstm_ol_mode$bacc[6] <- bacc(lstm_testB_mode$offensive_language, lstm_testB_mode$offensive_language_preds_A)
lstm_ol_mode$bacc[7] <- bacc(lstm_testB_mode$offensive_language, lstm_testB_mode$offensive_language_preds_B) 
lstm_ol_mode$bacc[8] <- bacc(lstm_testB_mode$offensive_language, lstm_testB_mode$offensive_language_preds_C)
lstm_ol_mode$bacc[9] <- bacc(lstm_testB_mode$offensive_language, lstm_testB_mode$offensive_language_preds_D)
lstm_ol_mode$bacc[10] <- bacc(lstm_testB_mode$offensive_language, lstm_testB_mode$offensive_language_preds_E)

lstm_ol_mode$bacc[11] <- bacc(lstm_testC_mode$offensive_language, lstm_testC_mode$offensive_language_preds_A)
lstm_ol_mode$bacc[12] <- bacc(lstm_testC_mode$offensive_language, lstm_testC_mode$offensive_language_preds_B) 
lstm_ol_mode$bacc[13] <- bacc(lstm_testC_mode$offensive_language, lstm_testC_mode$offensive_language_preds_C)
lstm_ol_mode$bacc[14] <- bacc(lstm_testC_mode$offensive_language, lstm_testC_mode$offensive_language_preds_D)
lstm_ol_mode$bacc[15] <- bacc(lstm_testC_mode$offensive_language, lstm_testC_mode$offensive_language_preds_E)

lstm_ol_mode$bacc[16] <- bacc(lstm_testD_mode$offensive_language, lstm_testD_mode$offensive_language_preds_A)
lstm_ol_mode$bacc[17] <- bacc(lstm_testD_mode$offensive_language, lstm_testD_mode$offensive_language_preds_B) 
lstm_ol_mode$bacc[18] <- bacc(lstm_testD_mode$offensive_language, lstm_testD_mode$offensive_language_preds_C)
lstm_ol_mode$bacc[19] <- bacc(lstm_testD_mode$offensive_language, lstm_testD_mode$offensive_language_preds_D)
lstm_ol_mode$bacc[20] <- bacc(lstm_testD_mode$offensive_language, lstm_testD_mode$offensive_language_preds_E)

lstm_ol_mode$bacc[21] <- bacc(lstm_testE_mode$offensive_language, lstm_testE_mode$offensive_language_preds_A)
lstm_ol_mode$bacc[22] <- bacc(lstm_testE_mode$offensive_language, lstm_testE_mode$offensive_language_preds_B) 
lstm_ol_mode$bacc[23] <- bacc(lstm_testE_mode$offensive_language, lstm_testE_mode$offensive_language_preds_C)
lstm_ol_mode$bacc[24] <- bacc(lstm_testE_mode$offensive_language, lstm_testE_mode$offensive_language_preds_D)
lstm_ol_mode$bacc[25] <- bacc(lstm_testE_mode$offensive_language, lstm_testE_mode$offensive_language_preds_E)

## Overall

sum_res <- data.frame(bert_ol = rep(NA, 5),
                      bert_hs = rep(NA, 5),
                      lstm_ol = rep(NA, 5),
                      lstm_hs = rep(NA, 5))

sum_res$bert_ol[1] <- auc(bert_test$offensive_language, bert_test$offensive.language_preds_A_scores, positive = "yes")
sum_res$bert_ol[2] <- auc(bert_test$offensive_language, bert_test$offensive.language_preds_B_scores, positive = "yes")
sum_res$bert_ol[3] <- auc(bert_test$offensive_language, bert_test$offensive.language_preds_C_scores, positive = "yes")
sum_res$bert_ol[4] <- auc(bert_test$offensive_language, bert_test$offensive.language_preds_D_scores, positive = "yes")
sum_res$bert_ol[5] <- auc(bert_test$offensive_language, bert_test$offensive.language_preds_E_scores, positive = "yes")

sum_res$bert_hs[1] <- auc(bert_test$hate_speech, bert_test$hate.speech_preds_A_scores, positive = "yes")
sum_res$bert_hs[2] <- auc(bert_test$hate_speech, bert_test$hate.speech_preds_B_scores, positive = "yes")
sum_res$bert_hs[3] <- auc(bert_test$hate_speech, bert_test$hate.speech_preds_C_scores, positive = "yes")
sum_res$bert_hs[4] <- auc(bert_test$hate_speech, bert_test$hate.speech_preds_D_scores, positive = "yes")
sum_res$bert_hs[5] <- auc(bert_test$hate_speech, bert_test$hate.speech_preds_E_scores, positive = "yes")

sum_res$lstm_ol[1] <- auc(lstm_test$offensive_language, lstm_test$offensive.language_preds_A_scores, positive = "yes")
sum_res$lstm_ol[2] <- auc(lstm_test$offensive_language, lstm_test$offensive.language_preds_B_scores, positive = "yes")
sum_res$lstm_ol[3] <- auc(lstm_test$offensive_language, lstm_test$offensive.language_preds_C_scores, positive = "yes")
sum_res$lstm_ol[4] <- auc(lstm_test$offensive_language, lstm_test$offensive.language_preds_D_scores, positive = "yes")
sum_res$lstm_ol[5] <- auc(lstm_test$offensive_language, lstm_test$offensive.language_preds_E_scores, positive = "yes")

sum_res$lstm_hs[1] <- auc(lstm_test$hate_speech, lstm_test$hate.speech_preds_A_scores, positive = "yes")
sum_res$lstm_hs[2] <- auc(lstm_test$hate_speech, lstm_test$hate.speech_preds_B_scores, positive = "yes")
sum_res$lstm_hs[3] <- auc(lstm_test$hate_speech, lstm_test$hate.speech_preds_C_scores, positive = "yes")
sum_res$lstm_hs[4] <- auc(lstm_test$hate_speech, lstm_test$hate.speech_preds_D_scores, positive = "yes")
sum_res$lstm_hs[5] <- auc(lstm_test$hate_speech, lstm_test$hate.speech_preds_E_scores, positive = "yes")
 
res_tex <- knitr::kable(sum_res, format = 'latex',  digits = 2)
writeLines(res_tex, 'res_tex.tex')

## Plots

ggplot(lstm_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.635, 0.735)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_hate_baccuracy_sampled.png", width = 6, height = 6)

ggplot(lstm_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.7, 0.8)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_offensive_baccuracy_sampled.png", width = 6, height = 6)

ggplot(lstm_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C", 
                      limits = c(0.71, 0.81)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_hate_auc_sampled.png", width = 6, height = 6)

ggplot(lstm_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3", 
                      limits = c(0.78, 0.88)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_offensive_auc_sampled.png", width = 6, height = 6)

ggplot(bert_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.665, 0.77)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_hate_baccuracy_sampled.png", width = 6, height = 6)

ggplot(bert_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.73, 0.835)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_offensive_baccuracy_sampled.png", width = 6, height = 6)

ggplot(bert_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.765, 0.87)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_hate_auc_sampled.png", width = 6, height = 6)

ggplot(bert_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.785, 0.89)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_offensive_auc_sampled.png", width = 6, height = 6)



ggplot(lstm_ol_mode, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2))) +
  scale_fill_gradient(low = "snow2", high = "palegreen4",
                      limits = c(0.7, 0.85)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

## Scatterplots

scaleFUN <- function(x) sprintf("%.1f", x)

lstm_testAs <- lstm_testA %>%
  group_by(version, tweet.id) %>%
  slice_sample(n = 1) %>%
  ungroup() 

bert_testAs <- bert_testA %>%
  group_by(version, tweet.id) %>%
  slice_sample(n = 1) %>%
  ungroup() 

lstm_testAs %>%
  select("P(HS) Train A" = "hate.speech_preds_A_scores",
         "P(HS) Train B" = "hate.speech_preds_B_scores",
         "P(HS) Train C" = "hate.speech_preds_C_scores",
         "P(HS) Train D" = "hate.speech_preds_D_scores",
         "P(HS) Train E" = "hate.speech_preds_E_scores") %>%
  ggpairs(lower = list(mapping = aes(color = "#F3941C", alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#F3941C')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("lstm_hate_scores_test_sampled.png", width = 9, height = 9)

lstm_testAs %>%
  select("P(OL) Train A" = "offensive.language_preds_A_scores",
         "P(OL) Train B" = "offensive.language_preds_B_scores",
         "P(OL) Train C" = "offensive.language_preds_C_scores",
         "P(OL) Train D" = "offensive.language_preds_D_scores",
         "P(OL) Train E" = "offensive.language_preds_E_scores") %>%
  ggpairs(lower = list(mapping = aes(color = "#009FE3", alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#009FE3')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("lstm_offensive_scores_test_sampled.png", width = 9, height = 9)

bert_testAs %>%
  select("P(HS) Train A" = "hate.speech_preds_A_scores",
         "P(HS) Train B" = "hate.speech_preds_B_scores",
         "P(HS) Train C" = "hate.speech_preds_C_scores",
         "P(HS) Train D" = "hate.speech_preds_D_scores",
         "P(HS) Train E" = "hate.speech_preds_E_scores") %>%
  ggpairs(lower = list(mapping = aes(color = "#F3941C", alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#F3941C')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("bert_hate_scores_test_sampled.png", width = 9, height = 9)

bert_testAs %>%
  select("P(OL) Train A" = "offensive.language_preds_A_scores",
         "P(OL) Train B" = "offensive.language_preds_B_scores",
         "P(OL) Train C" = "offensive.language_preds_C_scores",
         "P(OL) Train D" = "offensive.language_preds_D_scores",
         "P(OL) Train E" = "offensive.language_preds_E_scores") %>%
  ggpairs(lower = list(mapping = aes(color = "#009FE3", alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#009FE3')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("bert_offensive_scores_test_sampled.png", width = 9, height = 9)

