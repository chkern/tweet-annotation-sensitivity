##################
# ML Label Quality
# Learning Curves -- sampled
# R 3.6.3
##################

## Setup

library(tidyverse)
library(mlr3)
library(mlr3measures)
library(GGally)

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/preds_lstm_retrained/sampled")

lstm_testA <- read_csv(file = "lstm_testA_sampled_1to100.csv")
lstm_testB <- read_csv(file = "lstm_testB_sampled_1to100.csv")
lstm_testC <- read_csv(file = "lstm_testC_sampled_1to100.csv")
lstm_testD <- read_csv(file = "lstm_testD_sampled_1to100.csv")
lstm_testE <- read_csv(file = "lstm_testE_sampled_1to100.csv")

## 01 Prepare Data

lstm_testA <- lstm_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

lstm_testB <- lstm_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

lstm_testC <- lstm_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

lstm_testD <- lstm_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

lstm_testE <- lstm_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

lstm_test <- lstm_testA %>% 
  bind_rows(lstm_testB, lstm_testC, lstm_testD, lstm_testE) %>%
  drop_na()

## 02 Compare Classification Performance
### Hate Speech

hs_auc <- function(x) {
  return(auc(lstm_test$hate_speech, x, positive = "yes"))
}

hs_auc_A <- lstm_test %>%
  select(ends_with("speech_preds_A_scores")) %>%
  map_df(hs_auc)

hs_auc_B <- lstm_test %>%
  select(ends_with("speech_preds_B_scores")) %>%
  map_df(hs_auc)

hs_auc_C <- lstm_test %>%
  select(ends_with("speech_preds_C_scores")) %>%
  map_df(hs_auc)

hs_auc_D <- lstm_test %>%
  select(ends_with("speech_preds_D_scores")) %>%
  map_df(hs_auc)

hs_auc_E <- lstm_test %>%
  select(ends_with("speech_preds_E_scores")) %>%
  map_df(hs_auc)

hs_auc_res <- t(hs_auc_A) %>% 
  bind_cols(t(hs_auc_B), t(hs_auc_C), t(hs_auc_D), t(hs_auc_E))

names(hs_auc_res) <- c("A", "B", "C", "D", "E")

hs_auc_res <- hs_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

### Offensive Language

ol_auc <- function(x) {
  return(auc(lstm_test$offensive_language, x, positive = "yes"))
}

ol_auc_A <- lstm_test %>%
  select(ends_with("language_preds_A_scores")) %>%
  map_df(ol_auc)

ol_auc_B <- lstm_test %>%
  select(ends_with("language_preds_B_scores")) %>%
  map_df(ol_auc)

ol_auc_C <- lstm_test %>%
  select(ends_with("language_preds_C_scores")) %>%
  map_df(ol_auc)

ol_auc_D <- lstm_test %>%
  select(ends_with("language_preds_D_scores")) %>%
  map_df(ol_auc)

ol_auc_E <- lstm_test %>%
  select(ends_with("language_preds_E_scores")) %>%
  map_df(ol_auc)

ol_auc_res <- t(ol_auc_A) %>% 
  bind_cols(t(ol_auc_B), t(ol_auc_C), t(ol_auc_D), t(ol_auc_E))

names(ol_auc_res) <- c("A", "B", "C", "D", "E")

ol_auc_res <- ol_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

## Plots

ggplot(hs_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.5) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16))

ggsave("lstm_hate_curve_sampled.png", width = 9, height = 7)

ggplot(ol_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.5) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16))

ggsave("lstm_offensive_curve_sampled.png", width = 9, height = 7)
