##################
# ML Label Quality
# Prepare data for prediction modeling
# R 3.6.3
##################

## Setup

library(tidyverse)
library(vtable)
library(janitor)
library(readxl)
library(srvyr)

# tweets included in each batch
BATCH_SIZE = 50
NUM_TWEETS = 3000
NUM_BATCHES = NUM_TWEETS / BATCH_SIZE

dt0 <- read_excel("./analysis/final_export.xlsx") # labels
tweets <- read_csv("./tweets/stratified_sample_3000tweets.csv") # tweets

## 01: Data prep code from Stephanie

nrow(dt0)
tabyl(dt0, SHOWSECTION)

# remove curly brackets from var names
colnames(dt0) <- lapply(variable.names(dt0), str_remove_all, "[{}]")

dt1 <- dt0 %>% 
  rename(case.id = "Case ID") %>% 
  mutate(id = row_number()) %>% 
  rowwise %>% 
  # count number of tweets completed (at least 1st Q in version B,C,D,E)
  mutate(tweet.complete.ct = sum(!is.na(c_across(c(starts_with("TWTA"),
                                 starts_with("TWTB_Q1"),
                                 starts_with("TWTD_Q1")))))) %>% 
  mutate(complete = tweet.complete.ct == 20,
         # 2 ways to get version, make sure they match for completed cases (see below)
         version = factor(case_when(SHOWSECTION == 1 ~ "A",
                              SHOWSECTION == 2 ~ "B",
                              SHOWSECTION == 3 ~ "C",
                              SHOWSECTION == 4 ~ "D",
                              SHOWSECTION == 5 ~ "E"))) %>%
  # reduce to manageable columns
  select(id, version, starts_with("TWT"), starts_with("INDEX"))

review_subs <- c("62851efba924ed81f6f0e7f4", "5969496f06679c00014b818e", 
                 "5ce4eea02210eb0017870791", "616d09161d975b379aa79f99", 
                 "633af05da0499598ca6a9834", "62b674f342eb19fc64ed0646",
                 "60ad5f03939a57803959b569", "62851efba924ed81f6f0e7f4")
reviewed_subs <- dt0 %>% filter(PRL_ID %in% review_subs)

tabyl(dt1, version)

dt1.responses <- dt1 %>% 
  select(-starts_with("INDEX")) %>% 
  pivot_longer(c(starts_with("TWT")), 
               names_sep = "_",
               # tver is A, B, etc
               # question refers to 1st question about tweet, 2nd question about tweet
               # batch.tweet refers to 1st, 2nd, etc tweet in batch
               # (label itself is in var called value)
               # choice used in Version A only 1 -- HS; 2 -- OL; 3 -- Neither
               # see email from Jean 2022-11-17
               # each R does just one combination of these vars, will be lots of missing
               # but lets keep them for now in cases there are valid missing values
               names_to = c("tver", "question", "batch.tweet", "choice")) %>% 
  # only need the rows relevant for version respondent completed
  filter(substring(tver, 4) == version) %>% 
  select(-tver)

dt1.a <- dt1.responses %>% 
  filter(version == "A") %>%
  pivot_wider(id_cols = c(id, version, question, batch.tweet),
              names_from = choice,
              values_from = value) %>% 
  mutate(hate.speech = (C1 == 1),
         offensive.language = (C2 == 1))

# Q1 is HS; Q2 is OL -- check this
dt1.b <- dt1.responses %>% 
  filter(version == "B") %>% 
  pivot_wider(id_cols = c(id, version, batch.tweet),
              names_from = question,
              values_from = value) %>% 
  mutate(hate.speech = (Q1 == "1"),
         offensive.language = (Q2 == "1"))

# Q1 is OL; Q2 is HS -- check this
dt1.c <- dt1.responses %>% 
  filter(version == "C") %>% 
  pivot_wider(id_cols = c(id, version, batch.tweet),
              names_from = question,
              values_from = value) %>% 
  mutate(hate.speech = (Q2 == "1"),
         offensive.language = (Q1 == "1"))

# Q1 is HS; Q2 is OL -- check this
dt1.d <- dt1.responses %>% 
  filter(version == "D") %>% 
  pivot_wider(id_cols = c(id, version, batch.tweet),
              names_from = question,
              values_from = value) %>% 
  mutate(hate.speech = (Q1 == "1"),
         offensive.language = (Q2 == "1"))

# Q1 is OL; Q2 is HS -- check this
dt1.e <- dt1.responses %>% 
  filter(version == "E") %>% 
  pivot_wider(id_cols = c(id, version, batch.tweet),
              names_from = question,
              values_from = value) %>% 
  mutate(hate.speech = (Q2 == "1"),
         offensive.language = (Q1 == "1"))

dt1.all <- dt1.a %>% 
  bind_rows(dt1.b) %>% 
  bind_rows(dt1.c) %>% 
  bind_rows(dt1.d) %>% 
  bind_rows(dt1.e) %>% 
  select(-c(question, C1, C2, C3, Q1, Q2))
  
# INDEX* vars store ID of tweet 1-n
dt1.tweets <- dt1 %>% 
  # reduce to manageable columns
  select(id, version, starts_with("INDEX")) %>% 
  pivot_longer(c(starts_with("INDEX")), 
               names_sep = "_",
               # iver is A,B, etc
               # batch.tweet refers to 1st, 2nd, etc tweet in batch
               # (tweet.id itself is in var called value)
               # each R does just one combination of these vars, will be lots of missing
               # but lets keep them for now in cases there are valid missing values
               names_to = c("iver", "batch.tweet")) %>% 
  mutate(tweet.id = as.numeric(value)) %>% 
  filter(substring(iver, 6) == version) %>% 
  select(-iver)

dt2 <- left_join(dt1.all, 
                 dt1.tweets, 
                 by = c("id", "version", "batch.tweet")) 

dt3 <- dt2 %>% 
  filter(!is.na(tweet.id)) 

#############################
# QC

# number of annotators with 1 or more labeled tweets
nrow(dt3 %>% select(id) %>% distinct())

# number of labelled tweets per case
tabyl(dt3, id)

# number of labels per tweet & version
nrow(dt3 %>% distinct(tweet.id))
qc <- dt3 %>% 
  group_by(tweet.id, version) %>% 
  summarise(ct = n()) %>%
  ungroup() %>% 
  pivot_wider(names_from = version,
              values_from = ct)

# tweet with fewer than 3 codes
qc %>% 
  filter(A < 3 | B < 3 | C < 3 | D < 3 | E < 3)

# cases that did not finish
dt3 %>% 
  group_by(id) %>% 
  summarise(tweets.labelled = n()) %>% 
  filter(tweets.labelled != 50)

# HS, OL by version
tab <- dt3 %>% 
  group_by(version) %>% 
  summarise(mean.ol = mean(offensive.language, na.rm = TRUE),
            mean.hs = mean(hate.speech, na.rm = TRUE),
            mean.neither = mean(!hate.speech & !offensive.language, na.rm = TRUE),
            labels = n(),
            annotators = n_distinct(id)) %>% 
  arrange(desc(version))

tab_tex <- knitr::kable(tab, format = 'latex',  digits = 3)
writeLines(tab_tex, 'tab_tex.tex')

results <- dt3 %>% 
  as_survey_design(ids = id) %>% 
  group_by(version) %>% 
  summarise(mean_hs = survey_mean(hate.speech, na.rm = TRUE),
            mean_ol = survey_mean(offensive.language, na.rm = TRUE),
            mean_neither = survey_mean(!hate.speech & !offensive.language, na.rm = TRUE)) %>% 
  pivot_longer(starts_with("mean"),
               names_prefix = "mean_",
               names_to = "type") %>%
  separate(type, c("dv", "type2"), sep = "_") %>% 
  mutate(type = if_else(!is.na(type2), type2, "mean")) %>% 
  select(-type2) %>% 
  pivot_wider(names_from = type, 
              values_from = value)

results %>% 
  filter(dv %in% c("hs", "ol")) %>% 
  mutate(dv = recode_factor(dv, "ol" = "Offensive Language", "hs" = "Hate Speech")) %>%
  ggplot(aes(x = version, y = mean)) + 
  geom_bar(fill = "antiquewhite4", stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - 2*se, ymax = mean + 2*se), width = .2, position = position_dodge(.9)) +
  facet_wrap(. ~ dv) +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  ylim(c(0, 0.7)) +
  labs(x = "Version", y = "") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("desc.png", width = 8, height = 6)

# my_colors <- c("Black", "#00883A", "#DC0D15", "#3D4BE9", "#8C4091")
my_colors <- c("#009FE3", "#F3941C")

results %>% 
  filter(dv %in% c("hs", "ol")) %>% 
  mutate(dv = recode_factor(dv, "ol" = "Offensive Language", "hs" = "Hate Speech")) %>%
  ggplot(aes(x = version, y = mean, fill = dv)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - 2*se, ymax = mean + 2*se), width = .2, position = position_dodge(.9)) +
  facet_wrap(. ~ dv) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = my_colors) +
  coord_flip() +
  labs(x = "Condition", y = "") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("desc2.png", width = 8, height = 6)

#############################
## 02: Merge tweets to labels

only_tweets <- select(tweets, order, tweet_hashed)

dt4 <- dt3 %>%
  left_join(only_tweets, by = c("tweet.id" = "order")) %>% # double check
  mutate(hate.speech = as.numeric(hate.speech),
         offensive.language = as.numeric(offensive.language)) %>%
  select(-value) %>%
  arrange(version, tweet.id, id) %>%
  filter(!is.na(tweet_hashed)) %>%
  filter(!if_all(c(hate.speech, offensive.language), is.na))

summary(dt4)

## Three annotations per tweet and condition

dt4s <- dt4 %>% 
  group_by(version, tweet.id) %>%
  arrange(version, tweet.id, id) %>%
  slice_head(n = 3) %>%
  ungroup()

tab2 <- dt4s %>% 
  group_by(version) %>% 
  summarise(mean.ol = mean(offensive.language, na.rm = TRUE),
            mean.hs = mean(hate.speech, na.rm = TRUE),
            labels = n(),
            annotators = n_distinct(id)) %>% 
  arrange(desc(version))

tab2_tex <- knitr::kable(tab2, format = 'latex',  digits = 3)
writeLines(tab2_tex, 'tab2_tex.tex')

results2 <- dt4s %>% 
  as_survey_design(ids = id) %>% 
  group_by(version) %>% 
  summarise(mean_hs = survey_mean(hate.speech, na.rm = TRUE),
            mean_ol = survey_mean(offensive.language, na.rm = TRUE),
            mean_neither = survey_mean(!hate.speech & !offensive.language, na.rm = TRUE)) %>% 
  pivot_longer(starts_with("mean"),
               names_prefix = "mean_",
               names_to = "type") %>%
  separate(type, c("dv", "type2"), sep = "_") %>% 
  mutate(type = if_else(!is.na(type2), type2, "mean")) %>% 
  select(-type2) %>% 
  pivot_wider(names_from = type, 
              values_from = value)

# my_colors <- c("Black", "#00883A", "#DC0D15", "#3D4BE9", "#8C4091")
my_colors <- c("#009FE3", "#F3941C")

results2 %>% 
  filter(dv %in% c("hs", "ol")) %>% 
  mutate(dv = recode_factor(dv, "ol" = "Offensive Language", "hs" = "Hate Speech")) %>%
  ggplot(aes(x = version, y = mean, fill = dv)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean - 2*se, ymax = mean + 2*se), width = .2, position = position_dodge(.9)) +
  facet_wrap(. ~ dv) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = my_colors) +
  coord_flip() +
  labs(x = "Condition", y = "") + 
  theme(legend.position = "none",
        text = element_text(size = 16))

ggsave("desc3.png", width = 8, height = 6)

save(dt3, dt4, dt4s,
     file = "init_data.RData")
