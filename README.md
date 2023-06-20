## Overview

Publication: Annotation Collection Methods Affect Model Performance

Author(s): {author name(s)}{contact information}

Last update: June 20, 2023

Description: Machine learning (ML) training datasets often rely on human-annotated data, which are collected via online annotation instruments. However, design-driven bias and annotator effects can contribute to variation and noise in labels: even small changes in the annotation instrument can affect the collected annotations. This study builds on these results, demonstrating that the design of the annotation instrument also impacts the models trained on the resulting annotations. 

Using previously annotated Twitter data, we collect annotations of hate speech and offensive language in five experimental conditions of an annotation instrument, randomly assigning annotators to conditions. We then train LSTM and BERT models on each of the five resulting datasets and evaluate model performance on a holdout portion of each condition. We find considerable differences between the conditions for 1) the share of hate speech/offensive language annotations, 2) model performance, 3) model learning curves, and 4) model predictions. Our results emphasize the crucial role played by the annotation instrument which has received little attention in the machine learning literature. We call for additional research into how and why the collection instrument impacts the annotations collected to support the development of best practices in instrument design.

## Files and directories

1. Data pre-processing 
    + 00_setup.R
    + 01_split.R
    + 02_data_sampler.ipynb
    
2. Model training 
    + 03_lstm.ipynb
    + 04_bert.ipynb
    
3. Model evaluation
    + 05b_eval_sampled_acc.R
    + 05b_learn_sampled_acc.R

## Software

- R 3.6.3
- Python 3.6.4
