# Neural Fair Collaborative Filtering (NFCF)

This repository provides a simple demo for the NFCF model to debias career recommendations on the MovieLens data. 

## Prerequisites

* Python
* PyTorch

The code is tested on windows and linux operating systems. It should work on any other platform.

## Instructions

* Pre-processed MovieLens dataset is provided in "train-test" folder. The dataset contains interactions of users with movies and occupations.
* Baseline model: To fine tune NCF model without any fairness interventions, run the code using "run_fine_tuning_typical_ncf_career_recommend.py" file. 
**Training NFCF model:** 
* Step-1 (pre-training): Run the code using "run_preTrainNCF.py" file. The pre-trained NCF model will be saved in "trained-models" folder.\\
* Step-2 (de-biasing embeddings): Run the code using "run_debiasing_userEmbeddings.py" file. The de-biased user embeddings will be saved in "results" folder.\\
* Step-3 (fine-tuning): Run the code using "run_nfcf_career_recommend.py" file. The model will be fine-tuned with necessary fairness interventions. Evaluation results on the test set will be saved in the "results" folder.


## Author

* Rashidul Islam (email: islam.rashidul@umbc.edu)

## Reference Paper

[R. Islam, K.N. Keya, Z. Zeng, S. Pan, and J.R. Foulds. Debiasing career recommendations with neural fair collaborative filtering. In Proceedings of the Web Conference (WWW), 2021.](https://rashid-islam.github.io/homepage/files/papers/Debiasing_career_recommend_with_NFCF.pdf).

