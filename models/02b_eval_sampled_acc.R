##################
# ML Label Quality
# Evaluate models -- sampled (tuning: accuracy)
# R 3.6.3
##################

## Setup

library(tidyverse)
library(mlr3)
library(mlr3measures)
library(GGally)
library(gridExtra)

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/seed_lstm")

lstm_testA <- read_csv(file = "seed_lstm_testA_sampled.csv")
lstm_testB <- read_csv(file = "seed_lstm_testB_sampled.csv")
lstm_testC <- read_csv(file = "seed_lstm_testC_sampled.csv")
lstm_testD <- read_csv(file = "seed_lstm_testD_sampled.csv")
lstm_testE <- read_csv(file = "seed_lstm_testE_sampled.csv")

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/seed_bert")

bert_testA <- read_csv(file = "seed_bert_testA_sampled.csv")
bert_testB <- read_csv(file = "seed_bert_testB_sampled.csv")
bert_testC <- read_csv(file = "seed_bert_testC_sampled.csv")
bert_testD <- read_csv(file = "seed_bert_testD_sampled.csv")
bert_testE <- read_csv(file = "seed_bert_testE_sampled.csv")

## 01 Prepare Data

lstm_testA <- lstm_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

lstm_testB <- lstm_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

lstm_testC <- lstm_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

lstm_testD <- lstm_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes")) %>%
  drop_na(offensive_language)

lstm_testE <- lstm_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes")) %>%
  drop_na(hate_speech)

lstm_test <- lstm_testA %>% 
  bind_rows(lstm_testB, lstm_testC, lstm_testD, lstm_testE) %>%
  drop_na()

bert_testA <- bert_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

bert_testB <- bert_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

bert_testC <- bert_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes"))

bert_testD <- bert_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes")) %>%
  drop_na(offensive_language)

bert_testE <- bert_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language))) %>% 
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("speech_preds_A"),  ends_with("speech_preds_B"),
                 ends_with("speech_preds_C"), ends_with("speech_preds_D"),
                 ends_with("speech_preds_E")), factor, levels = c("no", "yes")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_at(vars(ends_with("language_preds_A"),  ends_with("language_preds_B"),
                 ends_with("language_preds_C"), ends_with("language_preds_D"),
                 ends_with("language_preds_E")), factor, levels = c("no", "yes")) %>%
  drop_na(hate_speech)

bert_test <- bert_testA %>% 
  bind_rows(bert_testB, bert_testC, bert_testD, bert_testE) %>%
  drop_na()

## 02 Compare Classification Performance
### Hate Speech

lstm_hate <- data.frame(bacc = rep(NA, 25),
                        acc = rep(NA, 25),
                        auc = rep(NA, 25),
                        test = rep(c("A", "B", "C", "D", "E"), each = 5),
                        train = rep(c("A", "B", "C", "D", "E"), 5))

hs_lstm_baccA <- function(x) {return(bacc(lstm_testA$hate_speech, x))}
hs_lstm_baccB <- function(x) {return(bacc(lstm_testB$hate_speech, x))}
hs_lstm_baccC <- function(x) {return(bacc(lstm_testC$hate_speech, x))}
hs_lstm_baccD <- function(x) {return(bacc(lstm_testD$hate_speech, x))}
hs_lstm_baccE <- function(x) {return(bacc(lstm_testE$hate_speech, x))}

lstm_hate$bacc[1] <- lstm_testA %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[2] <- lstm_testA %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[3] <- lstm_testA %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[4] <- lstm_testA %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[5] <- lstm_testA %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$bacc[6] <- lstm_testB %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[7] <- lstm_testB %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[8] <- lstm_testB %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[9] <- lstm_testB %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[10] <- lstm_testB %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$bacc[11] <- lstm_testC %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[12] <- lstm_testC %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[13] <- lstm_testC %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[14] <- lstm_testC %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[15] <- lstm_testC %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$bacc[16] <- lstm_testD %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[17] <- lstm_testD %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[18] <- lstm_testD %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[19] <- lstm_testD %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[20] <- lstm_testD %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$bacc[21] <- lstm_testE %>% select(ends_with("speech_preds_A")) %>%
  map_df(hs_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[22] <- lstm_testE %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[23] <- lstm_testE %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[24] <- lstm_testE %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$bacc[25] <- lstm_testE %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)

hs_lstm_accA <- function(x) {return(acc(lstm_testA$hate_speech, x))}
hs_lstm_accB <- function(x) {return(acc(lstm_testB$hate_speech, x))}
hs_lstm_accC <- function(x) {return(acc(lstm_testC$hate_speech, x))}
hs_lstm_accD <- function(x) {return(acc(lstm_testD$hate_speech, x))}
hs_lstm_accE <- function(x) {return(acc(lstm_testE$hate_speech, x))}

lstm_hate$acc[1] <- lstm_testA %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[2] <- lstm_testA %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[3] <- lstm_testA %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[4] <- lstm_testA %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[5] <- lstm_testA %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$acc[6] <- lstm_testB %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[7] <- lstm_testB %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[8] <- lstm_testB %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[9] <- lstm_testB %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[10] <- lstm_testB %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$acc[11] <- lstm_testC %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[12] <- lstm_testC %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[13] <- lstm_testC %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[14] <- lstm_testC %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[15] <- lstm_testC %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$acc[16] <- lstm_testD %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[17] <- lstm_testD %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[18] <- lstm_testD %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[19] <- lstm_testD %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[20] <- lstm_testD %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$acc[21] <- lstm_testE %>% select(ends_with("speech_preds_A")) %>%
  map_df(hs_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[22] <- lstm_testE %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[23] <- lstm_testE %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[24] <- lstm_testE %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$acc[25] <- lstm_testE %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)

hs_lstm_aucA <- function(x) {return(auc(lstm_testA$hate_speech, x, positive = "yes"))}
hs_lstm_aucB <- function(x) {return(auc(lstm_testB$hate_speech, x, positive = "yes"))}
hs_lstm_aucC <- function(x) {return(auc(lstm_testC$hate_speech, x, positive = "yes"))}
hs_lstm_aucD <- function(x) {return(auc(lstm_testD$hate_speech, x, positive = "yes"))}
hs_lstm_aucE <- function(x) {return(auc(lstm_testE$hate_speech, x, positive = "yes"))}

lstm_hate$auc[1] <- lstm_testA %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[2] <- lstm_testA %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[3] <- lstm_testA %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[4] <- lstm_testA %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[5] <- lstm_testA %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$auc[6] <- lstm_testB %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[7] <- lstm_testB %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[8] <- lstm_testB %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[9] <- lstm_testB %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[10] <- lstm_testB %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$auc[11] <- lstm_testC %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[12] <- lstm_testC %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[13] <- lstm_testC %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[14] <- lstm_testC %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[15] <- lstm_testC %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$auc[16] <- lstm_testD %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[17] <- lstm_testD %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[18] <- lstm_testD %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[19] <- lstm_testD %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[20] <- lstm_testD %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_hate$auc[21] <- lstm_testE %>% select(ends_with("speech_preds_A_scores")) %>%
  map_df(hs_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[22] <- lstm_testE %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[23] <- lstm_testE %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[24] <- lstm_testE %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_hate$auc[25] <- lstm_testE %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)

# lstm_f1_A_A <- fbeta(lstm_testA$hate_speech, lstm_testA$hate_speech_preds_A, positive = "yes")

bert_hate <- data.frame(bacc = rep(NA, 25),
                        acc = rep(NA, 25),
                        auc = rep(NA, 25),
                        test = rep(c("A", "B", "C", "D", "E"), each = 5),
                        train = rep(c("A", "B", "C", "D", "E"), 5))

hs_bert_baccA <- function(x) {return(bacc(bert_testA$hate_speech, x))}
hs_bert_baccB <- function(x) {return(bacc(bert_testB$hate_speech, x))}
hs_bert_baccC <- function(x) {return(bacc(bert_testC$hate_speech, x))}
hs_bert_baccD <- function(x) {return(bacc(bert_testD$hate_speech, x))}
hs_bert_baccE <- function(x) {return(bacc(bert_testE$hate_speech, x))}

bert_hate$bacc[1] <- bert_testA %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[2] <- bert_testA %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[3] <- bert_testA %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[4] <- bert_testA %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[5] <- bert_testA %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$bacc[6] <- bert_testB %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[7] <- bert_testB %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[8] <- bert_testB %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[9] <- bert_testB %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[10] <- bert_testB %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$bacc[11] <- bert_testC %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[12] <- bert_testC %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[13] <- bert_testC %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[14] <- bert_testC %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[15] <- bert_testC %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$bacc[16] <- bert_testD %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[17] <- bert_testD %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[18] <- bert_testD %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[19] <- bert_testD %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[20] <- bert_testD %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$bacc[21] <- bert_testE %>% select(ends_with("speech_preds_A")) %>%
  map_df(hs_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[22] <- bert_testE %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[23] <- bert_testE %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[24] <- bert_testE %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$bacc[25] <- bert_testE %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)

hs_bert_accA <- function(x) {return(acc(bert_testA$hate_speech, x))}
hs_bert_accB <- function(x) {return(acc(bert_testB$hate_speech, x))}
hs_bert_accC <- function(x) {return(acc(bert_testC$hate_speech, x))}
hs_bert_accD <- function(x) {return(acc(bert_testD$hate_speech, x))}
hs_bert_accE <- function(x) {return(acc(bert_testE$hate_speech, x))}

bert_hate$acc[1] <- bert_testA %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[2] <- bert_testA %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[3] <- bert_testA %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[4] <- bert_testA %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[5] <- bert_testA %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$acc[6] <- bert_testB %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[7] <- bert_testB %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[8] <- bert_testB %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[9] <- bert_testB %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[10] <- bert_testB %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$acc[11] <- bert_testC %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[12] <- bert_testC %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[13] <- bert_testC %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[14] <- bert_testC %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[15] <- bert_testC %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$acc[16] <- bert_testD %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[17] <- bert_testD %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[18] <- bert_testD %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[19] <- bert_testD %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[20] <- bert_testD %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$acc[21] <- bert_testE %>% select(ends_with("speech_preds_A")) %>%
  map_df(hs_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[22] <- bert_testE %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[23] <- bert_testE %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[24] <- bert_testE %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$acc[25] <- bert_testE %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)

hs_bert_aucA <- function(x) {return(auc(bert_testA$hate_speech, x, positive = "yes"))}
hs_bert_aucB <- function(x) {return(auc(bert_testB$hate_speech, x, positive = "yes"))}
hs_bert_aucC <- function(x) {return(auc(bert_testC$hate_speech, x, positive = "yes"))}
hs_bert_aucD <- function(x) {return(auc(bert_testD$hate_speech, x, positive = "yes"))}
hs_bert_aucE <- function(x) {return(auc(bert_testE$hate_speech, x, positive = "yes"))}

bert_hate$auc[1] <- bert_testA %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[2] <- bert_testA %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[3] <- bert_testA %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[4] <- bert_testA %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[5] <- bert_testA %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$auc[6] <- bert_testB %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[7] <- bert_testB %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[8] <- bert_testB %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[9] <- bert_testB %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[10] <- bert_testB %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$auc[11] <- bert_testC %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[12] <- bert_testC %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[13] <- bert_testC %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[14] <- bert_testC %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[15] <- bert_testC %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$auc[16] <- bert_testD %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[17] <- bert_testD %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[18] <- bert_testD %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[19] <- bert_testD %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[20] <- bert_testD %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_hate$auc[21] <- bert_testE %>% select(ends_with("speech_preds_A_scores")) %>%
  map_df(hs_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[22] <- bert_testE %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[23] <- bert_testE %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[24] <- bert_testE %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_hate$auc[25] <- bert_testE %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)

# bert_f1_A_A <- fbeta(bert_testA$hate_speech, bert_testA$hate_speech_preds_A, positive = "yes")

### Offensive Language

lstm_offensive <- data.frame(bacc = rep(NA, 25),
                             acc = rep(NA, 25),
                             auc = rep(NA, 25),
                             test = rep(c("A", "B", "C", "D", "E"), each = 5),
                             train = rep(c("A", "B", "C", "D", "E"), 5))

ol_lstm_baccA <- function(x) {return(bacc(lstm_testA$offensive_language, x))}
ol_lstm_baccB <- function(x) {return(bacc(lstm_testB$offensive_language, x))}
ol_lstm_baccC <- function(x) {return(bacc(lstm_testC$offensive_language, x))}
ol_lstm_baccD <- function(x) {return(bacc(lstm_testD$offensive_language, x))}
ol_lstm_baccE <- function(x) {return(bacc(lstm_testE$offensive_language, x))}

lstm_offensive$bacc[1] <- lstm_testA %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[2] <- lstm_testA %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[3] <- lstm_testA %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[4] <- lstm_testA %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[5] <- lstm_testA %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$bacc[6] <- lstm_testB %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[7] <- lstm_testB %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[8] <- lstm_testB %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[9] <- lstm_testB %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[10] <- lstm_testB %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$bacc[11] <- lstm_testC %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[12] <- lstm_testC %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[13] <- lstm_testC %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[14] <- lstm_testC %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[15] <- lstm_testC %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$bacc[16] <- lstm_testD %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[17] <- lstm_testD %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[18] <- lstm_testD %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[19] <- lstm_testD %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[20] <- lstm_testD %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$bacc[21] <- lstm_testE %>% select(ends_with("language_preds_A")) %>%
  map_df(ol_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[22] <- lstm_testE %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[23] <- lstm_testE %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[24] <- lstm_testE %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$bacc[25] <- lstm_testE %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_lstm_accA <- function(x) {return(acc(lstm_testA$offensive_language, x))}
ol_lstm_accB <- function(x) {return(acc(lstm_testB$offensive_language, x))}
ol_lstm_accC <- function(x) {return(acc(lstm_testC$offensive_language, x))}
ol_lstm_accD <- function(x) {return(acc(lstm_testD$offensive_language, x))}
ol_lstm_accE <- function(x) {return(acc(lstm_testE$offensive_language, x))}

lstm_offensive$acc[1] <- lstm_testA %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[2] <- lstm_testA %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[3] <- lstm_testA %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[4] <- lstm_testA %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[5] <- lstm_testA %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_accA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$acc[6] <- lstm_testB %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[7] <- lstm_testB %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[8] <- lstm_testB %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[9] <- lstm_testB %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[10] <- lstm_testB %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_accB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$acc[11] <- lstm_testC %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[12] <- lstm_testC %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[13] <- lstm_testC %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[14] <- lstm_testC %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[15] <- lstm_testC %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_accC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$acc[16] <- lstm_testD %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[17] <- lstm_testD %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[18] <- lstm_testD %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[19] <- lstm_testD %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[20] <- lstm_testD %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_accD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$acc[21] <- lstm_testE %>% select(ends_with("language_preds_A")) %>%
  map_df(ol_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[22] <- lstm_testE %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[23] <- lstm_testE %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[24] <- lstm_testE %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$acc[25] <- lstm_testE %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_accE) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_lstm_aucA <- function(x) {return(auc(lstm_testA$offensive_language, x, positive = "yes"))}
ol_lstm_aucB <- function(x) {return(auc(lstm_testB$offensive_language, x, positive = "yes"))}
ol_lstm_aucC <- function(x) {return(auc(lstm_testC$offensive_language, x, positive = "yes"))}
ol_lstm_aucD <- function(x) {return(auc(lstm_testD$offensive_language, x, positive = "yes"))}
ol_lstm_aucE <- function(x) {return(auc(lstm_testE$offensive_language, x, positive = "yes"))}

lstm_offensive$auc[1] <- lstm_testA %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[2] <- lstm_testA %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[3] <- lstm_testA %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[4] <- lstm_testA %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[5] <- lstm_testA %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$auc[6] <- lstm_testB %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[7] <- lstm_testB %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[8] <- lstm_testB %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[9] <- lstm_testB %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[10] <- lstm_testB %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$auc[11] <- lstm_testC %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[12] <- lstm_testC %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[13] <- lstm_testC %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[14] <- lstm_testC %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[15] <- lstm_testC %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$auc[16] <- lstm_testD %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[17] <- lstm_testD %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[18] <- lstm_testD %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[19] <- lstm_testD %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[20] <- lstm_testD %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)

lstm_offensive$auc[21] <- lstm_testE %>% select(ends_with("language_preds_A_scores")) %>%
  map_df(ol_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[22] <- lstm_testE %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[23] <- lstm_testE %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[24] <- lstm_testE %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
lstm_offensive$auc[25] <- lstm_testE %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)

# lstm_f1_A_A <- fbeta(lstm_testA$offensive_language, lstm_testA$offensive_language_preds_A, positive = "yes")

bert_offensive <- data.frame(bacc = rep(NA, 25),
                             acc = rep(NA, 25),
                             auc = rep(NA, 25),
                             test = rep(c("A", "B", "C", "D", "E"), each = 5),
                             train = rep(c("A", "B", "C", "D", "E"), 5))

ol_bert_baccA <- function(x) {return(bacc(bert_testA$offensive_language, x))}
ol_bert_baccB <- function(x) {return(bacc(bert_testB$offensive_language, x))}
ol_bert_baccC <- function(x) {return(bacc(bert_testC$offensive_language, x))}
ol_bert_baccD <- function(x) {return(bacc(bert_testD$offensive_language, x))}
ol_bert_baccE <- function(x) {return(bacc(bert_testE$offensive_language, x))}

bert_offensive$bacc[1] <- bert_testA %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[2] <- bert_testA %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[3] <- bert_testA %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[4] <- bert_testA %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[5] <- bert_testA %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_baccA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$bacc[6] <- bert_testB %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[7] <- bert_testB %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[8] <- bert_testB %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[9] <- bert_testB %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[10] <- bert_testB %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_baccB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$bacc[11] <- bert_testC %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[12] <- bert_testC %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[13] <- bert_testC %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[14] <- bert_testC %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[15] <- bert_testC %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_baccC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$bacc[16] <- bert_testD %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[17] <- bert_testD %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[18] <- bert_testD %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[19] <- bert_testD %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[20] <- bert_testD %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_baccD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$bacc[21] <- bert_testE %>% select(ends_with("language_preds_A")) %>%
  map_df(ol_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[22] <- bert_testE %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[23] <- bert_testE %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[24] <- bert_testE %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$bacc[25] <- bert_testE %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_baccE) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_bert_accA <- function(x) {return(acc(bert_testA$offensive_language, x))}
ol_bert_accB <- function(x) {return(acc(bert_testB$offensive_language, x))}
ol_bert_accC <- function(x) {return(acc(bert_testC$offensive_language, x))}
ol_bert_accD <- function(x) {return(acc(bert_testD$offensive_language, x))}
ol_bert_accE <- function(x) {return(acc(bert_testE$offensive_language, x))}

bert_offensive$acc[1] <- bert_testA %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[2] <- bert_testA %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[3] <- bert_testA %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[4] <- bert_testA %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[5] <- bert_testA %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_accA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$acc[6] <- bert_testB %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[7] <- bert_testB %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[8] <- bert_testB %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[9] <- bert_testB %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[10] <- bert_testB %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_accB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$acc[11] <- bert_testC %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[12] <- bert_testC %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[13] <- bert_testC %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[14] <- bert_testC %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[15] <- bert_testC %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_accC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$acc[16] <- bert_testD %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[17] <- bert_testD %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[18] <- bert_testD %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[19] <- bert_testD %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[20] <- bert_testD %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_accD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$acc[21] <- bert_testE %>% select(ends_with("language_preds_A")) %>%
  map_df(ol_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[22] <- bert_testE %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[23] <- bert_testE %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[24] <- bert_testE %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$acc[25] <- bert_testE %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_accE) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_bert_aucA <- function(x) {return(auc(bert_testA$offensive_language, x, positive = "yes"))}
ol_bert_aucB <- function(x) {return(auc(bert_testB$offensive_language, x, positive = "yes"))}
ol_bert_aucC <- function(x) {return(auc(bert_testC$offensive_language, x, positive = "yes"))}
ol_bert_aucD <- function(x) {return(auc(bert_testD$offensive_language, x, positive = "yes"))}
ol_bert_aucE <- function(x) {return(auc(bert_testE$offensive_language, x, positive = "yes"))}

bert_offensive$auc[1] <- bert_testA %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[2] <- bert_testA %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[3] <- bert_testA %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[4] <- bert_testA %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[5] <- bert_testA %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_aucA) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$auc[6] <- bert_testB %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[7] <- bert_testB %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[8] <- bert_testB %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[9] <- bert_testB %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[10] <- bert_testB %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_aucB) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$auc[11] <- bert_testC %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[12] <- bert_testC %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[13] <- bert_testC %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[14] <- bert_testC %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[15] <- bert_testC %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_aucC) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$auc[16] <- bert_testD %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[17] <- bert_testD %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[18] <- bert_testD %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[19] <- bert_testD %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[20] <- bert_testD %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_aucD) %>% mutate(m = rowMeans(.)) %>% select(m)

bert_offensive$auc[21] <- bert_testE %>% select(ends_with("language_preds_A_scores")) %>%
  map_df(ol_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[22] <- bert_testE %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[23] <- bert_testE %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[24] <- bert_testE %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)
bert_offensive$auc[25] <- bert_testE %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_aucE) %>% mutate(m = rowMeans(.)) %>% select(m)

# bert_f1_A_A <- fbeta(bert_testA$offensive_language, bert_testA$offensive_language_preds_A, positive = "yes")

## Overall

sum_auc <- data.frame(bert_ol = rep(NA, 5),
                      bert_hs = rep(NA, 5),
                      lstm_ol = rep(NA, 5),
                      lstm_hs = rep(NA, 5))

ol_lstm_auc <- function(x) {return(auc(lstm_test$offensive_language, x, positive = "yes"))}
hs_lstm_auc <- function(x) {return(auc(lstm_test$hate_speech, x, positive = "yes"))}

sum_auc$lstm_ol[1] <- lstm_test %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_ol[2] <- lstm_test %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_ol[3] <- lstm_test %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_ol[4] <- lstm_test %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_ol[5] <- lstm_test %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_auc$lstm_hs[1] <- lstm_test %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_hs[2] <- lstm_test %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_hs[3] <- lstm_test %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_hs[4] <- lstm_test %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$lstm_hs[5] <- lstm_test %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_lstm_auc) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_bert_auc <- function(x) {return(auc(bert_test$offensive_language, x, positive = "yes"))}
hs_bert_auc <- function(x) {return(auc(bert_test$hate_speech, x, positive = "yes"))}

sum_auc$bert_ol[1] <- bert_test %>% select(ends_with("language_preds_A_scores")) %>% 
  map_df(ol_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_ol[2] <- bert_test %>% select(ends_with("language_preds_B_scores")) %>% 
  map_df(ol_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_ol[3] <- bert_test %>% select(ends_with("language_preds_C_scores")) %>% 
  map_df(ol_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_ol[4] <- bert_test %>% select(ends_with("language_preds_D_scores")) %>% 
  map_df(ol_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_ol[5] <- bert_test %>% select(ends_with("language_preds_E_scores")) %>% 
  map_df(ol_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_auc$bert_hs[1] <- bert_test %>% select(ends_with("speech_preds_A_scores")) %>% 
  map_df(hs_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_hs[2] <- bert_test %>% select(ends_with("speech_preds_B_scores")) %>% 
  map_df(hs_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_hs[3] <- bert_test %>% select(ends_with("speech_preds_C_scores")) %>% 
  map_df(hs_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_hs[4] <- bert_test %>% select(ends_with("speech_preds_D_scores")) %>% 
  map_df(hs_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_auc$bert_hs[5] <- bert_test %>% select(ends_with("speech_preds_E_scores")) %>% 
  map_df(hs_bert_auc) %>% mutate(m = rowMeans(.)) %>% select(m)

res1_tex <- knitr::kable(sum_auc, format = 'latex',  digits = 2)
writeLines(res1_tex, 'res_acc_auc.tex')

sum_bacc <- data.frame(bert_ol = rep(NA, 5),
                       bert_hs = rep(NA, 5),
                       lstm_ol = rep(NA, 5),
                       lstm_hs = rep(NA, 5))

ol_lstm_bacc <- function(x) {return(bacc(lstm_test$offensive_language, x))}
hs_lstm_bacc <- function(x) {return(bacc(lstm_test$hate_speech, x))}

sum_bacc$lstm_ol[1] <- lstm_test %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_ol[2] <- lstm_test %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_ol[3] <- lstm_test %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_ol[4] <- lstm_test %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_ol[5] <- lstm_test %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_bacc$lstm_hs[1] <- lstm_test %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_hs[2] <- lstm_test %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_hs[3] <- lstm_test %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_hs[4] <- lstm_test %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$lstm_hs[5] <- lstm_test %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_bert_bacc <- function(x) {return(bacc(bert_test$offensive_language, x))}
hs_bert_bacc <- function(x) {return(bacc(bert_test$hate_speech, x))}

sum_bacc$bert_ol[1] <- bert_test %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_ol[2] <- bert_test %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_ol[3] <- bert_test %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_ol[4] <- bert_test %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_ol[5] <- bert_test %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_bacc$bert_hs[1] <- bert_test %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_hs[2] <- bert_test %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_hs[3] <- bert_test %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_hs[4] <- bert_test %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_bacc$bert_hs[5] <- bert_test %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_bacc) %>% mutate(m = rowMeans(.)) %>% select(m)

res2_tex <- knitr::kable(sum_bacc, format = 'latex',  digits = 2)
writeLines(res2_tex, 'res_acc_bacc.tex')

sum_acc <- data.frame(bert_ol = rep(NA, 5),
                      bert_hs = rep(NA, 5),
                      lstm_ol = rep(NA, 5),
                      lstm_hs = rep(NA, 5))

ol_lstm_acc <- function(x) {return(acc(lstm_test$offensive_language, x))}
hs_lstm_acc <- function(x) {return(acc(lstm_test$hate_speech, x))}

sum_acc$lstm_ol[1] <- lstm_test %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_ol[2] <- lstm_test %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_ol[3] <- lstm_test %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_ol[4] <- lstm_test %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_ol[5] <- lstm_test %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_acc$lstm_hs[1] <- lstm_test %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_hs[2] <- lstm_test %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_hs[3] <- lstm_test %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_hs[4] <- lstm_test %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$lstm_hs[5] <- lstm_test %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_lstm_acc) %>% mutate(m = rowMeans(.)) %>% select(m)

ol_bert_acc <- function(x) {return(acc(bert_test$offensive_language, x))}
hs_bert_acc <- function(x) {return(acc(bert_test$hate_speech, x))}

sum_acc$bert_ol[1] <- bert_test %>% select(ends_with("language_preds_A")) %>% 
  map_df(ol_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_ol[2] <- bert_test %>% select(ends_with("language_preds_B")) %>% 
  map_df(ol_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_ol[3] <- bert_test %>% select(ends_with("language_preds_C")) %>% 
  map_df(ol_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_ol[4] <- bert_test %>% select(ends_with("language_preds_D")) %>% 
  map_df(ol_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_ol[5] <- bert_test %>% select(ends_with("language_preds_E")) %>% 
  map_df(ol_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)

sum_acc$bert_hs[1] <- bert_test %>% select(ends_with("speech_preds_A")) %>% 
  map_df(hs_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_hs[2] <- bert_test %>% select(ends_with("speech_preds_B")) %>% 
  map_df(hs_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_hs[3] <- bert_test %>% select(ends_with("speech_preds_C")) %>% 
  map_df(hs_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_hs[4] <- bert_test %>% select(ends_with("speech_preds_D")) %>% 
  map_df(hs_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)
sum_acc$bert_hs[5] <- bert_test %>% select(ends_with("speech_preds_E")) %>% 
  map_df(hs_bert_acc) %>% mutate(m = rowMeans(.)) %>% select(m)

res3_tex <- knitr::kable(sum_acc, format = 'latex',  digits = 2)
writeLines(res3_tex, 'res_acc_acc.tex')

## Plots

# data.frame(lapply(lstm_hate, unlist))

ggplot(lstm_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = acc)) + 
  geom_text(aes(label = round(acc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.72, 0.82)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_hate_sampled_acc_acc.png", width = 6, height = 6)

ggplot(lstm_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = acc)) + 
  geom_text(aes(label = round(acc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.7, 0.8)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_offensive_sampled_acc_acc.png", width = 6, height = 6)

ggplot(lstm_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.635, 0.735)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_hate_sampled_acc_bacc.png", width = 6, height = 6)

ggplot(lstm_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.7, 0.8)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_offensive_sampled_acc_bacc.png", width = 6, height = 6)

ggplot(lstm_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C", 
                      limits = c(0.7, 0.8)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_hate_sampled_acc_auc.png", width = 6, height = 6)

ggplot(lstm_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3", 
                      limits = c(0.78, 0.88)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("lstm_offensive_sampled_acc_auc.png", width = 6, height = 6)

ggplot(bert_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = acc)) + 
  geom_text(aes(label = round(acc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.745, 0.845)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_hate_sampled_acc_acc.png", width = 6, height = 6)

ggplot(bert_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = acc)) + 
  geom_text(aes(label = round(acc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.745, 0.845)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_offensive_sampled_acc_acc.png", width = 6, height = 6)

ggplot(bert_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.63, 0.75)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_hate_sampled_acc_bacc.png", width = 6, height = 6)

ggplot(bert_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = bacc)) + 
  geom_text(aes(label = round(bacc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.74, 0.84)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_offensive_sampled_acc_bacc.png", width = 6, height = 6)

ggplot(bert_hate, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#F3941C",
                      limits = c(0.77, 0.87)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_hate_sampled_acc_auc.png", width = 6, height = 6)

ggplot(bert_offensive, aes(x = test, y = fct_rev(train))) + 
  geom_raster(aes(fill = auc)) + 
  geom_text(aes(label = round(auc, 2)), size = 5) +
  scale_fill_gradient(low = "snow2", high = "#009FE3",
                      limits = c(0.8, 0.9)) +
  labs(x = "Test", y = "Train") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("bert_offensive_sampled_acc_auc.png", width = 6, height = 6)


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
  ggpairs(lower = list(continuous = wrap("points", color = "#F3941C", 
                                         size = 1, stroke = 0.2, alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#F3941C')) +
  scale_size_manual(0.0001) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("lstm_hate_sampled_acc_scores.png", width = 9, height = 9)

lstm_testAs %>%
  select("P(OL) Train A" = "offensive.language_preds_A_scores",
         "P(OL) Train B" = "offensive.language_preds_B_scores",
         "P(OL) Train C" = "offensive.language_preds_C_scores",
         "P(OL) Train D" = "offensive.language_preds_D_scores",
         "P(OL) Train E" = "offensive.language_preds_E_scores") %>%
  ggpairs(lower = list(continuous = wrap("points", color = "#009FE3", 
                                         size = 1, stroke = 0.2, alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#009FE3')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("lstm_offensive_sampled_acc_scores.png", width = 9, height = 9)

bert_testAs %>%
  select("P(HS) Train A" = "hate.speech_preds_A_scores",
         "P(HS) Train B" = "hate.speech_preds_B_scores",
         "P(HS) Train C" = "hate.speech_preds_C_scores",
         "P(HS) Train D" = "hate.speech_preds_D_scores",
         "P(HS) Train E" = "hate.speech_preds_E_scores") %>%
  ggpairs(lower = list(continuous = wrap("points", color = "#F3941C", 
                                         position = position_jitter(w = 0.02, h = 0.01),
                                         size = 1, stroke = 0.2, alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#F3941C')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("bert_hate_sampled_acc_scores.png", width = 9, height = 9)

bert_testAs %>%
  select("P(OL) Train A" = "offensive.language_preds_A_scores",
         "P(OL) Train B" = "offensive.language_preds_B_scores",
         "P(OL) Train C" = "offensive.language_preds_C_scores",
         "P(OL) Train D" = "offensive.language_preds_D_scores",
         "P(OL) Train E" = "offensive.language_preds_E_scores") %>%
  ggpairs(lower = list(continuous = wrap("points", color = "#009FE3", 
                                         position = position_jitter(w = 0.02, h = 0.01),
                                         size = 1, stroke = 0.2, alpha = 0.5)),
          upper = list(continuous = wrap("cor", size = 7))) +
  scale_x_continuous(labels = scaleFUN) +
  scale_y_continuous(labels = scaleFUN) +
  scale_colour_manual(values = c('#009FE3')) +
  theme(text = element_text(size = 16),
        axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   vjust = 1))

ggsave("bert_offensive_sampled_acc_scores.png", width = 9, height = 9)

## Difference in scores

lstm_testAsdiff <- lstm_testAs %>%
  mutate(hate.speech_AB = abs(hate.speech_preds_A_scores - hate.speech_preds_B_scores),
         hate.speech_AC = abs(hate.speech_preds_A_scores - hate.speech_preds_C_scores),
         hate.speech_AD = abs(hate.speech_preds_A_scores - hate.speech_preds_D_scores),
         hate.speech_AE = abs(hate.speech_preds_A_scores - hate.speech_preds_E_scores),
         hate.speech_BC = abs(hate.speech_preds_B_scores - hate.speech_preds_C_scores),
         hate.speech_BD = abs(hate.speech_preds_B_scores - hate.speech_preds_D_scores),
         hate.speech_BE = abs(hate.speech_preds_B_scores - hate.speech_preds_E_scores),
         hate.speech_CD = abs(hate.speech_preds_C_scores - hate.speech_preds_D_scores),
         hate.speech_CE = abs(hate.speech_preds_C_scores - hate.speech_preds_E_scores),
         hate.speech_DE = abs(hate.speech_preds_D_scores - hate.speech_preds_E_scores)) %>%
  mutate(offensive.language_AB = abs(offensive.language_preds_A_scores - offensive.language_preds_B_scores),
         offensive.language_AC = abs(offensive.language_preds_A_scores - offensive.language_preds_C_scores),
         offensive.language_AD = abs(offensive.language_preds_A_scores - offensive.language_preds_D_scores),
         offensive.language_AE = abs(offensive.language_preds_A_scores - offensive.language_preds_E_scores),
         offensive.language_BC = abs(offensive.language_preds_B_scores - offensive.language_preds_C_scores),
         offensive.language_BD = abs(offensive.language_preds_B_scores - offensive.language_preds_D_scores),
         offensive.language_BE = abs(offensive.language_preds_B_scores - offensive.language_preds_E_scores),
         offensive.language_CD = abs(offensive.language_preds_C_scores - offensive.language_preds_D_scores),
         offensive.language_CE = abs(offensive.language_preds_C_scores - offensive.language_preds_E_scores),
         offensive.language_DE = abs(offensive.language_preds_D_scores - offensive.language_preds_E_scores))

lstm_diff <- lstm_testAsdiff %>%
  select(hate.speech_AB:hate.speech_DE, offensive.language_AB:offensive.language_DE) %>%
  pivot_longer(everything(),
               names_to = c("outcome", "diff"),
               names_sep = "_")

bert_testAsdiff <- bert_testAs %>%
  mutate(hate.speech_AB = abs(hate.speech_preds_A_scores - hate.speech_preds_B_scores),
         hate.speech_AC = abs(hate.speech_preds_A_scores - hate.speech_preds_C_scores),
         hate.speech_AD = abs(hate.speech_preds_A_scores - hate.speech_preds_D_scores),
         hate.speech_AE = abs(hate.speech_preds_A_scores - hate.speech_preds_E_scores),
         hate.speech_BC = abs(hate.speech_preds_B_scores - hate.speech_preds_C_scores),
         hate.speech_BD = abs(hate.speech_preds_B_scores - hate.speech_preds_D_scores),
         hate.speech_BE = abs(hate.speech_preds_B_scores - hate.speech_preds_E_scores),
         hate.speech_CD = abs(hate.speech_preds_C_scores - hate.speech_preds_D_scores),
         hate.speech_CE = abs(hate.speech_preds_C_scores - hate.speech_preds_E_scores),
         hate.speech_DE = abs(hate.speech_preds_D_scores - hate.speech_preds_E_scores)) %>%
  mutate(offensive.language_AB = abs(offensive.language_preds_A_scores - offensive.language_preds_B_scores),
         offensive.language_AC = abs(offensive.language_preds_A_scores - offensive.language_preds_C_scores),
         offensive.language_AD = abs(offensive.language_preds_A_scores - offensive.language_preds_D_scores),
         offensive.language_AE = abs(offensive.language_preds_A_scores - offensive.language_preds_E_scores),
         offensive.language_BC = abs(offensive.language_preds_B_scores - offensive.language_preds_C_scores),
         offensive.language_BD = abs(offensive.language_preds_B_scores - offensive.language_preds_D_scores),
         offensive.language_BE = abs(offensive.language_preds_B_scores - offensive.language_preds_E_scores),
         offensive.language_CD = abs(offensive.language_preds_C_scores - offensive.language_preds_D_scores),
         offensive.language_CE = abs(offensive.language_preds_C_scores - offensive.language_preds_E_scores),
         offensive.language_DE = abs(offensive.language_preds_D_scores - offensive.language_preds_E_scores))

bert_diff <- bert_testAsdiff %>%
  select(hate.speech_AB:hate.speech_DE, offensive.language_AB:offensive.language_DE) %>%
  pivot_longer(everything(),
               names_to = c("outcome", "diff"),
               names_sep = "_")

## Density Plots

l_hs_A <- lstm_testAs %>%
  ggplot(aes(x = hate.speech_preds_A_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - A") +
  theme(text = element_text(size = 14))

l_hs_B <- lstm_testAs %>%
  ggplot(aes(x = hate.speech_preds_B_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - B") +
  theme(text = element_text(size = 14))

l_hs_C <- lstm_testAs %>%
  ggplot(aes(x = hate.speech_preds_C_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - C") +
  theme(text = element_text(size = 14))

l_hs_D <- lstm_testAs %>%
  ggplot(aes(x = hate.speech_preds_D_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - D") +
  theme(text = element_text(size = 14))

l_hs_E <- lstm_testAs %>%
  ggplot(aes(x = hate.speech_preds_E_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - E") +
  theme(text = element_text(size = 14))

l_ol_A <- lstm_testAs %>%
  ggplot(aes(x = offensive.language_preds_A_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - A") +
  theme(text = element_text(size = 14))

l_ol_B <- lstm_testAs %>%
  ggplot(aes(x = offensive.language_preds_B_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - B") +
  theme(text = element_text(size = 14))

l_ol_C <- lstm_testAs %>%
  ggplot(aes(x = offensive.language_preds_C_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - C") +
  theme(text = element_text(size = 14))

l_ol_D <- lstm_testAs %>%
  ggplot(aes(x = offensive.language_preds_D_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - D") +
  theme(text = element_text(size = 14))

l_ol_E <- lstm_testAs %>%
  ggplot(aes(x = offensive.language_preds_E_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - E") +
  theme(text = element_text(size = 14))

b_hs_A <- bert_testAs %>%
  ggplot(aes(x = hate.speech_preds_A_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - A") +
  theme(text = element_text(size = 14))

b_hs_B <- bert_testAs %>%
  ggplot(aes(x = hate.speech_preds_B_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - B") +
  theme(text = element_text(size = 14))

b_hs_C <- bert_testAs %>%
  ggplot(aes(x = hate.speech_preds_C_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - C") +
  theme(text = element_text(size = 14))

b_hs_D <- bert_testAs %>%
  ggplot(aes(x = hate.speech_preds_D_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - D") +
  theme(text = element_text(size = 14))

b_hs_E <- bert_testAs %>%
  ggplot(aes(x = hate.speech_preds_E_scores)) +
  geom_density(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - E") +
  theme(text = element_text(size = 14))

b_ol_A <- bert_testAs %>%
  ggplot(aes(x = offensive.language_preds_A_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - A") +
  theme(text = element_text(size = 14))

b_ol_B <- bert_testAs %>%
  ggplot(aes(x = offensive.language_preds_B_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - B") +
  theme(text = element_text(size = 14))

b_ol_C <- bert_testAs %>%
  ggplot(aes(x = offensive.language_preds_C_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - C") +
  theme(text = element_text(size = 14))

b_ol_D <- bert_testAs %>%
  ggplot(aes(x = offensive.language_preds_D_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - D") +
  theme(text = element_text(size = 14))

b_ol_E <- bert_testAs %>%
  ggplot(aes(x = offensive.language_preds_E_scores)) +
  geom_density(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  ggtitle("Density - E") +
  theme(text = element_text(size = 14))

## Difference Plots 

l_hs_AB <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "AB") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("A vs. B") +
  theme(text = element_text(size = 14))

l_hs_AC <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "AC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("A vs. C") +
  theme(text = element_text(size = 14))

l_hs_AD <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "AD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("A vs. D") +
  theme(text = element_text(size = 14))

l_hs_AE <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "AE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("A vs. E") +
  theme(text = element_text(size = 14))

l_hs_BC <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "BC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("B vs. C") +
  theme(text = element_text(size = 14))

l_hs_BD <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "BD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("B vs. D") +
  theme(text = element_text(size = 14))

l_hs_BE <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "BE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("B vs. E") +
  theme(text = element_text(size = 14))

l_hs_CD <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "CD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("C vs. D") +
  theme(text = element_text(size = 14))

l_hs_CE <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "CE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("C vs. E") +
  theme(text = element_text(size = 14))

l_hs_DE <- lstm_diff %>%
  filter(outcome == "hate.speech" & diff == "DE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 185)) +
  ggtitle("D vs. E") +
  theme(text = element_text(size = 14))

g <- grid.arrange(l_hs_A,
                  l_hs_AB, l_hs_B,
                  l_hs_AC, l_hs_BC, l_hs_C,
                  l_hs_AD, l_hs_BD, l_hs_CD, l_hs_D,
                  l_hs_AE, l_hs_BE, l_hs_CE, l_hs_DE, l_hs_E,
                  layout_matrix = rbind(c(1, NA, NA, NA, NA),
                                        c(2, 3,  NA, NA, NA),
                                        c(4, 5, 6, NA, NA),
                                        c(7, 8, 9, 10, NA),
                                        c(11, 12, 13, 14, 15)))

ggsave("lstm_hate_sampled_acc_diff.png", g, width = 9, height = 9)

l_ol_AB <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "AB") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("A vs. B") +
  theme(text = element_text(size = 14))

l_ol_AC <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "AC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("A vs. C") +
  theme(text = element_text(size = 14))

l_ol_AD <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "AD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("A vs. D") +
  theme(text = element_text(size = 14))

l_ol_AE <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "AE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("A vs. E") +
  theme(text = element_text(size = 14))

l_ol_BC <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "BC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("B vs. C") +
  theme(text = element_text(size = 14))

l_ol_BD <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "BD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("B vs. D") +
  theme(text = element_text(size = 14))

l_ol_BE <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "BE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("B vs. E") +
  theme(text = element_text(size = 14))

l_ol_CD <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "CD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("C vs. D") +
  theme(text = element_text(size = 14))

l_ol_CE <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "CE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("C vs. E") +
  theme(text = element_text(size = 14))

l_ol_DE <- lstm_diff %>%
  filter(outcome == "offensive.language" & diff == "DE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 200)) +
  ggtitle("D vs. E") +
  theme(text = element_text(size = 14))

g <- grid.arrange(l_ol_A,
                  l_ol_AB, l_ol_B,
                  l_ol_AC, l_ol_BC, l_ol_C,
                  l_ol_AD, l_ol_BD, l_ol_CD, l_ol_D,
                  l_ol_AE, l_ol_BE, l_ol_CE, l_ol_DE, l_ol_E,
                  layout_matrix = rbind(c(1, NA, NA, NA, NA),
                                        c(2, 3,  NA, NA, NA),
                                        c(4, 5, 6, NA, NA),
                                        c(7, 8, 9, 10, NA),
                                        c(11, 12, 13, 14, 15)))

ggsave("lstm_offensive_sampled_acc_diff.png", g, width = 9, height = 9)

b_hs_AB <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "AB") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("A vs. B") +
  theme(text = element_text(size = 14))

b_hs_AC <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "AC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("A vs. C") +
  theme(text = element_text(size = 14))

b_hs_AD <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "AD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("A vs. D") +
  theme(text = element_text(size = 14))

b_hs_AE <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "AE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("A vs. E") +
  theme(text = element_text(size = 14))

b_hs_BC <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "BC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("B vs. C") +
  theme(text = element_text(size = 14))

b_hs_BD <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "BD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("B vs. D") +
  theme(text = element_text(size = 14))

b_hs_BE <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "BE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("B vs. E") +
  theme(text = element_text(size = 14))

b_hs_CD <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "CD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("C vs. D") +
  theme(text = element_text(size = 14))

b_hs_CE <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "CE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("C vs. E") +
  theme(text = element_text(size = 14))

b_hs_DE <- bert_diff %>%
  filter(outcome == "hate.speech" & diff == "DE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#F3941C") +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 275)) +
  ggtitle("D vs. E") +
  theme(text = element_text(size = 14))

g <- grid.arrange(b_hs_A,
                  b_hs_AB, b_hs_B,
                  b_hs_AC, b_hs_BC, b_hs_C,
                  b_hs_AD, b_hs_BD, b_hs_CD, b_hs_D,
                  b_hs_AE, b_hs_BE, b_hs_CE, b_hs_DE, b_hs_E,
                  layout_matrix = rbind(c(1, NA, NA, NA, NA),
                                        c(2, 3,  NA, NA, NA),
                                        c(4, 5, 6, NA, NA),
                                        c(7, 8, 9, 10, NA),
                                        c(11, 12, 13, 14, 15)))

ggsave("bert_hate_sampled_acc_diff.png", g, width = 9, height = 9)

b_ol_AB <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "AB") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("A vs. B") +
  theme(text = element_text(size = 14))

b_ol_AC <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "AC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("A vs. C") +
  theme(text = element_text(size = 14))

b_ol_AD <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "AD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("A vs. D") +
  theme(text = element_text(size = 14))

b_ol_AE <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "AE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("A vs. E") +
  theme(text = element_text(size = 14))

b_ol_BC <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "BC") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("B vs. C") +
  theme(text = element_text(size = 14))

b_ol_BD <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "BD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("B vs. D") +
  theme(text = element_text(size = 14))

b_ol_BE <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "BE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("B vs. E") +
  theme(text = element_text(size = 14))

b_ol_CD <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "CD") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("C vs. D") +
  theme(text = element_text(size = 14))

b_ol_CE <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "CE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("C vs. E") +
  theme(text = element_text(size = 14))

b_ol_DE <- bert_diff %>%
  filter(outcome == "offensive.language" & diff == "DE") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "#009FE3", bins = 45) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(ylim = c(0, 325)) +
  ggtitle("D vs. E") +
  theme(text = element_text(size = 14))

g <- grid.arrange(b_ol_A,
                  b_ol_AB, b_ol_B,
                  b_ol_AC, b_ol_BC, b_ol_C,
                  b_ol_AD, b_ol_BD, b_ol_CD, b_ol_D,
                  b_ol_AE, b_ol_BE, b_ol_CE, b_ol_DE, b_ol_E,
                  layout_matrix = rbind(c(1, NA, NA, NA, NA),
                                        c(2, 3,  NA, NA, NA),
                                        c(4, 5, 6, NA, NA),
                                        c(7, 8, 9, 10, NA),
                                        c(11, 12, 13, 14, 15)))

ggsave("bert_offensive_sampled_acc_diff.png", g, width = 9, height = 9)
