# Data

Twitter data annotated in five experimental conditions (A-E). 

The data was randomly split into a train (75%) and a test (25%) set.

The full dataset is available at Huggingface: https://huggingface.co/datasets/soda-lmu/tweet-annotation-sensitivity-2.

| Column Name     | Description       |
| -------------- | ------------------ |
| id  | annotator ID |
| version  | experimental condition (A-E) |
| batch.tweet  | tweet id in batch |
| hate.speech  | binary label for the class "hate speech", <br>0 for "non hate speech", 1 for "hate speech"|
| offensive.language  | binary label for the class "offensive language", <br>0 for "non offensive language", 1 for "offensive language" |
| tweet.id  | tweet id |
| tweet_hashed  | tweet |



