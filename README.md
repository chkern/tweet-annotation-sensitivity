## Overview

Publication: Annotation Sensitivity: Training Data Collection Methods Affect Model Performance

Author(s): {author name(s)}{contact information}

Last update: June 23, 2023

Description: Machine learning training datasets often rely on human-annotated data, which are collected via online annotation instruments. However, the design of the annotation instrument, the instructions given to annotators, the characteristics of the annotators, and their interactions can contribute to noise in training data, a phenomenon we call annotation sensitivity. Even small changes in the annotation instrument can affect the collected annotations. This study demonstrates that the design of the annotation instrument also impacts the models trained on the resulting annotations. 

Using previously annotated Twitter data, we collect annotations of hate speech and offensive language in five experimental conditions of an annotation instrument, randomly assigning annotators to conditions. We then train LSTM and BERT models on each of the five resulting datasets and evaluate model performance on a holdout portion of each condition. We find considerable differences between the conditions for 1) the share of hate speech/offensive language annotations, 2) model performance, 3) model learning curves, and 4) model predictions. 

Our results emphasize the crucial role played by the annotation instrument which has received little attention in the machine learning literature. We call for additional research into how and why the collection instrument impacts the annotations collected to support the development of best practices in instrument design. 

## Files and directories

1. Data pre-processing 
    + 00_setup.R
    + 01_split.R
    
2. Model training 
    + 02_lstm.ipynb
    + 03_bert.ipynb
    
3. Model evaluation
    + 04b_eval_sampled_acc.R
    + 04b_learn_sampled_acc.R

4. Data used in this paper
    + 05_data
      + full train and text data
      + split train and text data for each version (A-E)
      + train_dev_split for model training 

## Software

- R (3.6.3)
  - tidyverse (1.3.1)
  - readxl (1.3.1)
  - janitor (2.1.0)
  - srvyr (1.0.1)
  - mlr3 (0.13.3)
  - mlr3measures (0.4.1)

- Python (3.10.6)
  - nltk (3.8.1)
  - numpy (1.22.2)
  - pandas (1.5.2)
  - torch (2.0.0)
  - tqdm (4.65.0)
  - transformers (4.28.1)
