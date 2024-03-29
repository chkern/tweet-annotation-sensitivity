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

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/preds_lstm_sampled_1to100")

lstm_testA <- read_csv(file = "lstm_testA_sampled_1to100.csv")
lstm_testB <- read_csv(file = "lstm_testB_sampled_1to100.csv")
lstm_testC <- read_csv(file = "lstm_testC_sampled_1to100.csv")
lstm_testD <- read_csv(file = "lstm_testD_sampled_1to100.csv")
lstm_testE <- read_csv(file = "lstm_testE_sampled_1to100.csv")

setwd("~/Uni/Forschung/Article/2022 - LabelQuali/src/preds_bert_sampled_1to100")

bert_testA <- read_csv(file = "bert_testA_sampled_1to100.csv")
bert_testB <- read_csv(file = "bert_testB_sampled_1to100.csv")
bert_testC <- read_csv(file = "bert_testC_sampled_1to100.csv")
bert_testD <- read_csv(file = "bert_testD_sampled_1to100.csv")
bert_testE <- read_csv(file = "bert_testE_sampled_1to100.csv")

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

bert_testA <- bert_testA %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

bert_testB <- bert_testB %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

bert_testC <- bert_testC %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

bert_testD <- bert_testD %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

bert_testE <- bert_testE %>%
  mutate(hate_speech = factor(ifelse(hate.speech == 1, "yes", "no")),
         hate_speech = na_if(hate_speech, is.na(hate.speech)),
         offensive_language = factor(ifelse(offensive.language == 1, "yes", "no")),
         offensive_language = na_if(offensive_language, is.na(offensive.language)))

bert_test <- bert_testA %>% 
  bind_rows(bert_testB, bert_testC, bert_testD, bert_testE) %>%
  drop_na()

## 02 Compare Classification Performance
### Hate Speech -- ROC-AUC

hs_lstm_auc <- function(x) {
  return(auc(lstm_test$hate_speech, x, positive = "yes"))
}

hs_bert_auc <- function(x) {
  return(auc(bert_test$hate_speech, x, positive = "yes"))
}

hs_lstm_auc_A <- lstm_test %>%
  select(ends_with("speech_preds_A_scores")) %>%
  map_df(hs_lstm_auc)

hs_lstm_auc_B <- lstm_test %>%
  select(ends_with("speech_preds_B_scores")) %>%
  map_df(hs_lstm_auc)

hs_lstm_auc_C <- lstm_test %>%
  select(ends_with("speech_preds_C_scores")) %>%
  map_df(hs_lstm_auc)

hs_lstm_auc_D <- lstm_test %>%
  select(ends_with("speech_preds_D_scores")) %>%
  map_df(hs_lstm_auc)

hs_lstm_auc_E <- lstm_test %>%
  select(ends_with("speech_preds_E_scores")) %>%
  map_df(hs_lstm_auc)

hs_lstm_auc_res <- t(hs_lstm_auc_A) %>% 
  bind_cols(t(hs_lstm_auc_B), t(hs_lstm_auc_C), t(hs_lstm_auc_D), t(hs_lstm_auc_E))

names(hs_lstm_auc_res) <- c("A", "B", "C", "D", "E")

hs_lstm_auc_res <- hs_lstm_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

hs_bert_auc_A <- bert_test %>%
  select(ends_with("speech_preds_A_scores")) %>%
  map_df(hs_bert_auc)

hs_bert_auc_B <- bert_test %>%
  select(ends_with("speech_preds_B_scores")) %>%
  map_df(hs_bert_auc)

hs_bert_auc_C <- bert_test %>%
  select(ends_with("speech_preds_C_scores")) %>%
  map_df(hs_bert_auc)

hs_bert_auc_D <- bert_test %>%
  select(ends_with("speech_preds_D_scores")) %>%
  map_df(hs_bert_auc)

hs_bert_auc_E <- bert_test %>%
  select(ends_with("speech_preds_E_scores")) %>%
  map_df(hs_bert_auc)

hs_bert_auc_res <- t(hs_bert_auc_A) %>% 
  bind_cols(t(hs_bert_auc_B), t(hs_bert_auc_C), t(hs_bert_auc_D), t(hs_bert_auc_E))

names(hs_bert_auc_res) <- c("A", "B", "C", "D", "E")

hs_bert_auc_res <- hs_bert_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

### Hate Speech -- bacc

hs_lstm_bacc <- function(x) {
  return(bacc(lstm_test$hate_speech, x))
}

hs_bert_bacc <- function(x) {
  return(bacc(bert_test$hate_speech, x))
}

hs_lstm_bacc_A <- lstm_test %>%
  select(ends_with("speech_preds_A")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_lstm_bacc)

hs_lstm_bacc_B <- lstm_test %>%
  select(ends_with("speech_preds_B")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_lstm_bacc)

hs_lstm_bacc_C <- lstm_test %>%
  select(ends_with("speech_preds_C")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_lstm_bacc)

hs_lstm_bacc_D <- lstm_test %>%
  select(ends_with("speech_preds_D")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_lstm_bacc)

hs_lstm_bacc_E <- lstm_test %>%
  select(ends_with("speech_preds_E")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_lstm_bacc)

hs_lstm_bacc_res <- t(hs_lstm_bacc_A) %>% 
  bind_cols(t(hs_lstm_bacc_B), t(hs_lstm_bacc_C), t(hs_lstm_bacc_D), t(hs_lstm_bacc_E))

names(hs_lstm_bacc_res) <- c("A", "B", "C", "D", "E")

hs_lstm_bacc_res <- hs_lstm_bacc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "bacc")

hs_bert_bacc_A <- bert_test %>%
  select(ends_with("speech_preds_A")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_bert_bacc)

hs_bert_bacc_B <- bert_test %>%
  select(ends_with("speech_preds_B")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_bert_bacc)

hs_bert_bacc_C <- bert_test %>%
  select(ends_with("speech_preds_C")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_bert_bacc)

hs_bert_bacc_D <- bert_test %>%
  select(ends_with("speech_preds_D")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_bert_bacc)

hs_bert_bacc_E <- bert_test %>%
  select(ends_with("speech_preds_E")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(hs_bert_bacc)

hs_bert_bacc_res <- t(hs_bert_bacc_A) %>% 
  bind_cols(t(hs_bert_bacc_B), t(hs_bert_bacc_C), t(hs_bert_bacc_D), t(hs_bert_bacc_E))

names(hs_bert_bacc_res) <- c("A", "B", "C", "D", "E")

hs_bert_bacc_res <- hs_bert_bacc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "bacc")

### Offensive Language -- ROC-AUC

ol_lstm_auc <- function(x) {
  return(auc(lstm_test$offensive_language, x, positive = "yes"))
}

ol_bert_auc <- function(x) {
  return(auc(bert_test$offensive_language, x, positive = "yes"))
}

ol_lstm_auc_A <- lstm_test %>%
  select(ends_with("language_preds_A_scores")) %>%
  map_df(ol_lstm_auc)

ol_lstm_auc_B <- lstm_test %>%
  select(ends_with("language_preds_B_scores")) %>%
  map_df(ol_lstm_auc)

ol_lstm_auc_C <- lstm_test %>%
  select(ends_with("language_preds_C_scores")) %>%
  map_df(ol_lstm_auc)

ol_lstm_auc_D <- lstm_test %>%
  select(ends_with("language_preds_D_scores")) %>%
  map_df(ol_lstm_auc)

ol_lstm_auc_E <- lstm_test %>%
  select(ends_with("language_preds_E_scores")) %>%
  map_df(ol_lstm_auc)

ol_lstm_auc_res <- t(ol_lstm_auc_A) %>% 
  bind_cols(t(ol_lstm_auc_B), t(ol_lstm_auc_C), t(ol_lstm_auc_D), t(ol_lstm_auc_E))

names(ol_lstm_auc_res) <- c("A", "B", "C", "D", "E")

ol_lstm_auc_res <- ol_lstm_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

ol_bert_auc_A <- bert_test %>%
  select(ends_with("language_preds_A_scores")) %>%
  map_df(ol_bert_auc)

ol_bert_auc_B <- bert_test %>%
  select(ends_with("language_preds_B_scores")) %>%
  map_df(ol_bert_auc)

ol_bert_auc_C <- bert_test %>%
  select(ends_with("language_preds_C_scores")) %>%
  map_df(ol_bert_auc)

ol_bert_auc_D <- bert_test %>%
  select(ends_with("language_preds_D_scores")) %>%
  map_df(ol_bert_auc)

ol_bert_auc_E <- bert_test %>%
  select(ends_with("language_preds_E_scores")) %>%
  map_df(ol_bert_auc)

ol_bert_auc_res <- t(ol_bert_auc_A) %>% 
  bind_cols(t(ol_bert_auc_B), t(ol_bert_auc_C), t(ol_bert_auc_D), t(ol_bert_auc_E))

names(ol_bert_auc_res) <- c("A", "B", "C", "D", "E")

ol_bert_auc_res <- ol_bert_auc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "AUC")

### Offensive Language -- bacc

ol_lstm_bacc <- function(x) {
  return(bacc(lstm_test$offensive_language, x))
}

ol_bert_bacc <- function(x) {
  return(bacc(bert_test$offensive_language, x))
}

ol_lstm_bacc_A <- lstm_test %>%
  select(ends_with("language_preds_A")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_lstm_bacc)

ol_lstm_bacc_B <- lstm_test %>%
  select(ends_with("language_preds_B")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_lstm_bacc)

ol_lstm_bacc_C <- lstm_test %>%
  select(ends_with("language_preds_C")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_lstm_bacc)

ol_lstm_bacc_D <- lstm_test %>%
  select(ends_with("language_preds_D")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_lstm_bacc)

ol_lstm_bacc_E <- lstm_test %>%
  select(ends_with("language_preds_E")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_lstm_bacc)

ol_lstm_bacc_res <- t(ol_lstm_bacc_A) %>% 
  bind_cols(t(ol_lstm_bacc_B), t(ol_lstm_bacc_C), t(ol_lstm_bacc_D), t(ol_lstm_bacc_E))

names(ol_lstm_bacc_res) <- c("A", "B", "C", "D", "E")

ol_lstm_bacc_res <- ol_lstm_bacc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "bacc")

ol_bert_bacc_A <- bert_test %>%
  select(ends_with("language_preds_A")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_bert_bacc)

ol_bert_bacc_B <- bert_test %>%
  select(ends_with("language_preds_B")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_bert_bacc)

ol_bert_bacc_C <- bert_test %>%
  select(ends_with("language_preds_C")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_bert_bacc)

ol_bert_bacc_D <- bert_test %>%
  select(ends_with("language_preds_D")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_bert_bacc)

ol_bert_bacc_E <- bert_test %>%
  select(ends_with("language_preds_E")) %>%
  mutate_all(function(x) ifelse(x == 1, "yes", "no")) %>%
  mutate_all(factor, levels = c("no", "yes")) %>%
  map_df(ol_bert_bacc)

ol_bert_bacc_res <- t(ol_bert_bacc_A) %>% 
  bind_cols(t(ol_bert_bacc_B), t(ol_bert_bacc_C), t(ol_bert_bacc_D), t(ol_bert_bacc_E))

names(ol_bert_bacc_res) <- c("A", "B", "C", "D", "E")

ol_bert_bacc_res <- ol_bert_bacc_res %>% 
  rowid_to_column("iter") %>%
  pivot_longer(!iter, names_to = "Condition", values_to = "bacc")

## Plots

my_colors <- c("#8C4091", "#3D4BE9", "#DC0D15", "#00883A", "#000000")

ggplot(hs_lstm_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("lstm_hate_curve_sampled_acc_auc.png", width = 7, height = 7)

ggplot(ol_lstm_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("lstm_offensive_curve_sampled_acc_auc.png", width = 7, height = 7)

ggplot(hs_bert_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("bert_hate_curve_sampled_acc_auc.png", width = 7, height = 7)

ggplot(ol_bert_auc_res, aes(x = iter, y = AUC, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Test Set ROC-AUC", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("bert_offensive_curve_sampled_acc_auc.png", width = 7, height = 7)

ggplot(hs_lstm_bacc_res, aes(x = iter, y = bacc, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Bal. Accuracy", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("lstm_hate_curve_sampled_acc_bacc.png", width = 7, height = 7)

ggplot(ol_lstm_bacc_res, aes(x = iter, y = bacc, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Bal. Accuracy", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("lstm_offensive_curve_sampled_acc_bacc.png", width = 7, height = 7)

ggplot(hs_bert_bacc_res, aes(x = iter, y = bacc, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Bal. Accuracy", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("bert_hate_curve_sampled_acc_bacc.png", width = 7, height = 7)

ggplot(ol_bert_bacc_res, aes(x = iter, y = bacc, group = Condition)) + 
  geom_line(aes(color = Condition), alpha = 0.4) + 
  geom_smooth(aes(color = Condition), fill = "lightgray") + 
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = my_colors) +
  labs(x = "Relative Size of Training Data", y = "Bal. Accuracy", color = "Training\nCondition") + 
  theme(text = element_text(size = 16),
        legend.position = "bottom")

ggsave("bert_offensive_curve_sampled_acc_bacc.png", width = 7, height = 7)
