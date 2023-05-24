##################
# ML Label Quality
# Prepare data for prediction modeling
# R 3.6.3
##################

## Setup

library(tidyverse)

load("init_data.RData")

##################
## Data splits -- All annotations

set.seed(923436)

full_train <- dt4 %>% 
  group_by(tweet.id) %>%
  nest() %>%                    # sample on tweet level
  ungroup() %>%
  slice_sample(prop = 0.75) %>% # 75% of any version
  unnest()

full_test <- dt4 %>% 
  anti_join(full_train, by = "tweet.id") # 25% for test sets

versionA_train <- full_train %>% filter(version == "A")
versionA_test <- full_test %>% filter(version == "A")
nrow(versionA_train %>% distinct(tweet.id)) # 2250
nrow(versionA_test %>% distinct(tweet.id)) # 750

versionB_train <- full_train %>% filter(version == "B")
versionB_test <- full_test %>% filter(version == "B")
nrow(versionB_train %>% distinct(tweet.id))
nrow(versionB_test %>% distinct(tweet.id))

versionC_train <- full_train %>% filter(version == "C")
versionC_test <- full_test %>% filter(version == "C")
nrow(versionC_train %>% distinct(tweet.id))
nrow(versionC_test %>% distinct(tweet.id))

versionD_train <- full_train %>% filter(version == "D")
versionD_test <- full_test %>% filter(version == "D")
nrow(versionD_train %>% distinct(tweet.id))
nrow(versionD_test %>% distinct(tweet.id))

versionE_train <- full_train %>% filter(version == "E")
versionE_test <- full_test %>% filter(version == "E")
nrow(versionE_train %>% distinct(tweet.id))
nrow(versionE_test %>% distinct(tweet.id))

save(full_train, full_test,
     versionA_train, versionA_test,
     versionB_train, versionB_test,
     versionC_train, versionC_test,
     versionD_train, versionD_test,
     versionE_train, versionE_test,
     file = "ml_data.RData")

write_csv(full_train, path = "full_train.csv")
write_csv(full_test, path = "full_test.csv")

write_csv(versionA_train, path = "versionA_train.csv")
write_csv(versionA_test, path = "versionA_test.csv")

write_csv(versionB_train, path = "versionB_train.csv")
write_csv(versionB_test, path = "versionB_test.csv")

write_csv(versionC_train, path = "versionC_train.csv")
write_csv(versionC_test, path = "versionC_test.csv")

write_csv(versionD_train, path = "versionD_train.csv")
write_csv(versionD_test, path = "versionD_test.csv")

write_csv(versionE_train, path = "versionE_train.csv")
write_csv(versionE_test, path = "versionE_test.csv")

##################
## Data splits -- Three annotations per tweet and condition

set.seed(923436)

full_train_s <- dt4s %>% 
  group_by(tweet.id) %>%
  nest() %>%                    # sample on tweet level
  ungroup() %>%
  slice_sample(prop = 0.75) %>% # 75% of any version
  unnest()

full_test_s <- dt4s %>% 
  anti_join(full_train_s, by = "tweet.id") # 25% for test sets

versionA_train_s <- full_train_s %>% filter(version == "A")
versionA_test_s <- full_test_s %>% filter(version == "A")
nrow(versionA_train_s %>% distinct(tweet.id)) # 2250
nrow(versionA_test_s %>% distinct(tweet.id)) # 750

versionB_train_s <- full_train_s %>% filter(version == "B")
versionB_test_s <- full_test_s %>% filter(version == "B")

versionC_train_s <- full_train_s %>% filter(version == "C")
versionC_test_s <- full_test_s %>% filter(version == "C")

versionD_train_s <- full_train_s %>% filter(version == "D")
versionD_test_s <- full_test_s %>% filter(version == "D")

versionE_train_s <- full_train_s %>% filter(version == "E")
versionE_test_s <- full_test_s %>% filter(version == "E")

save(full_train_s, full_test_s,
     versionA_train_s, versionA_test_s,
     versionB_train_s, versionB_test_s,
     versionC_train_s, versionC_test_s,
     versionD_train_s, versionD_test_s,
     versionE_train_s, versionE_test_s,
     file = "ml_data_s.RData")

write_csv(full_train_s, path = "full_train_s.csv")
write_csv(full_test_s, path = "full_test_s.csv")

write_csv(versionA_train_s, path = "versionA_train_s.csv")
write_csv(versionA_test_s, path = "versionA_test_s.csv")

write_csv(versionB_train_s, path = "versionB_train_s.csv")
write_csv(versionB_test_s, path = "versionB_test_s.csv")

write_csv(versionC_train_s, path = "versionC_train_s.csv")
write_csv(versionC_test_s, path = "versionC_test_s.csv")

write_csv(versionD_train_s, path = "versionD_train_s.csv")
write_csv(versionD_test_s, path = "versionD_test_s.csv")

write_csv(versionE_train_s, path = "versionE_train_s.csv")
write_csv(versionE_test_s, path = "versionE_test_s.csv")
