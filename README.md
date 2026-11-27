# Cycling Results Analysis and Top-20 Prediction

University of Pisa Data Mining group project. The project combines historical road cycling race results with rider and race attributes to explore patterns, predict top-20 finishes, and explain model predictions. The [project report](report.pdf) describes the full analysis.

## Workflow

1. [Data understanding](ipynb_files/1_data_understanding.ipynb) explores the race and rider tables.
2. [Data transformation](ipynb_files/2_data_transformation.ipynb) cleans the records and builds features from earlier seasons, including historical rider placement.
3. [Clustering](ipynb_files/3_data-clustering.ipynb) explores groups of riders and races.
4. [Prediction](ipynb_files/4_data_prediction.ipynb) compares classifiers and resampling strategies for the imbalanced top-20 target. Races before 2022 were used for development, and races from 2022 onward were held out for testing.
5. [Explainability](ipynb_files/5_xai.ipynb) compares TreeSHAP settings on one tree from the bagging model and explores local explanations with LIME.

## Saved test results

The test set contains 23,132 rider–race records, of which 3,464 are top-20 finishes. Precision, recall and F1 refer to the top-20 class. Each row represents a saved model and resampling configuration. [Saved evaluation records](ipynb_files/logs/best_models.json)

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 73.37% | 0.25 | 0.38 | 0.30 |
| k-nearest neighbors | 63.34% | 0.23 | 0.60 | 0.33 |
| Multilayer perceptron | 69.33% | 0.28 | 0.69 | 0.40 |
| Bagging | 74.29% | 0.31 | 0.61 | 0.41 |
| AdaBoost | 66.27% | 0.25 | 0.65 | 0.36 |

Bagging had the highest top-20 F1 among the saved configurations. The MLP caught more top-20 finishes and had lower precision. Predicting outside the top 20 for every row would yield 85.03% accuracy and zero top-20 recall, which is why the positive-class metrics matter here.

Training data was resampled before cross-validation in parts of the notebook. To verify model selection scores, a follow-up would resample within each training fold and check that all features are available before race start.

## Running the notebooks

Set up a Python environment with the pinned [dependencies](requirements.txt) and open the notebooks in Jupyter. Start with data understanding and transformation, then run the clustering and prediction notebooks before explainability. Review the dataset paths in each notebook before execution. The saved results can be inspected directly without retraining.
