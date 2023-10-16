## Overview

#### Annotation Sensitivity: Training Data Collection Methods Affect Model Performance**

**Abstract:**

Machine learning training datasets often rely on human-annotated data, which are collected via online annotation instruments. However, the design of the annotation instrument, the instructions given to annotators, the characteristics of the annotators, and their interactions can contribute to noise in training data, a phenomenon we call annotation sensitivity. Even small changes in the annotation instrument can affect the collected annotations. This study demonstrates that the design of the annotation instrument also impacts the models trained on the resulting annotations. 

Using previously annotated Twitter data, we collect annotations of hate speech and offensive language in five experimental conditions of an annotation instrument, randomly assigning annotators to conditions. We then train LSTM and BERT models on each of the five resulting datasets and evaluate model performance on a holdout portion of each condition. We find considerable differences between the conditions for 1) the share of hate speech/offensive language annotations, 2) model performance, 3) model learning curves, and 4) model predictions. 

Our results emphasize the crucial role played by the annotation instrument which has received little attention in the machine learning literature. We call for additional research into how and why the collection instrument impacts the annotations collected to support the development of best practices in instrument design. 

**Annotation Method:**

We conducted tweet data annotations of **hate speech** (HS) and **offensive language** (OL) in five experimental conditions. The tweet data was sampled from the corpus created by [Davidson et al. (2017)](https://ojs.aaai.org/index.php/ICWSM/article/view/14955). We selected 3,000 Tweets for our annotation. We developed five experimental conditions that varied the annotation task structure, as shown in the following figure. All tweets were annotated in each condition.

- **<font color= #871F78>Condition A</font>** presented the tweet and three options on a single screen: hate speech, offensive language, or neither. Annotators could select one or both of hate speech, offensive language, or indicate that neither applied.

- Conditions B and C split the annotation of a single tweet across two screens.
  + For **<font color= Blue>Condition B</font>**, the first screen prompted the annotator to indicate whether the tweet contained hate speech. On the following screen, they were shown the tweet again and asked whether it contained offensive language.
  + **<font color= red>Condition C</font>** was similar to Condition B, but flipped the order of hate speech and offensive language for each tweet. 

- In Conditions D and E, the two tasks are treated independently with annotators being asked to first annotate all tweets for one task, followed by annotating all tweets again for the second task.
  + Annotators assigned **<font color=green>Condition D</font>** were first asked to annotate hate speech for all their assigned tweets, and then asked to annotate offensive language for the same set of tweets.
  + **Condition E** worked the same way, but started with the offensive language annotation task followed by the hate speech annotation task.  

The full dataset is available at Huggingface: https://huggingface.co/datasets/soda-lmu/tweet-annotation-sensitivity-2

<br />

<img src="https://raw.githubusercontent.com/chkern/tweet-annotation-sensitivity/main/fig/exp_conditions.png" width = "300" height = "450" alt="" align=center />


## Files and directories

- ``data``: 
    - Train und test data
      + `full_train_s.csv`
      + `full_test_s.csv`
  

- ``models``:

  - Data pre-processing 
      + `00_setup.R`
      + `01_split.R`
      
  - Model training 
      + `02_lstm.ipynb`
      + `03_bert.ipynb`
      
  - Model evaluation
      + `04b_eval_sampled_acc.R`
      + `04b_learn_sampled_acc.R`


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

## Citation

If you find this repository useful, please cite:
```
@inproceedings{kern-et-al-2023-annotation,
    title = "Annotation Sensitivity: Training Data Collection Methods Affect Model Performance",
    author = "Christoph Kern and Stephanie Eckman and Jacob Beck and Rob Chew and Bolei Ma and Frauke Kreuter",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = "dec",
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics"
}
```
