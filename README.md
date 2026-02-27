# Rock-Paper-Scissors CNN Classification project

## Report
See the project report here: [ML_report_GV.pdf](ML_report_GV.pdf)

## Overview
This project develops and evaluates Convolutional Neural Networks (CNNs) to accurately classify hand gestures corresponding to the Rock-Paper-Scissors game. The work was carried out as part of the Machine Learning course assignment, with the objective of following sound statistical and machine learning practices across the full pipeline, from data processing to model evaluation. 

The project prioritizes methodological rigor over raw accuracy, implementing proper data splitting, cross-validation and automated hyperparameter tuning, while keeping training time reasonable given the dataset size and task complexity. 

The dataset used is [Rock-Paper-Scissors dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) downloaded from kaggle containing 2188 images across 3 balanced classes (paper, rock, scissors), all photographed against a green screen background.

## Project structure
```
/rock_paper_scissors
|
├── data/                          # Dataset directory (not included in repository)
│   ├── paper/
│   ├── rock/
│   └── scissors/
|
├── models/                        # Saved model files (not included in repository)
│   ├── model_1_baseline.keras
│   ├── model_2_intermediate.keras
│   ├── model_3_advanced.keras
│   └── model_2_tuned_final.keras
|
├── my_test_data/                  # Personal generalization test images (not included for privacy)
|
├── .vscode/                       # VS Code settings
├── rps.py                         # Python script version of the full pipeline
├── rps.ipynb                      # Jupyter Notebook version with outputs and visualizations
├── ML_report_GV.pdf               # Project report
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## Data Preprocessing
Before any model training, the dataset was carefully preprocessed to ensure zero data leakage and full reproducibility. The images were first split at the file level into training (70%), validation (15%), and test (15%) sets using a fixed random seed (42), guaranteeing that no information from the validation or test sets could influence the training pipeline.

Each image was then resized to 64x64 pixels and normalized to the [0, 1] range by rescaling pixel values by 1/255. To improve generalization and increase dataset diversity, data augmentation was applied exclusively to the training set, including:

- Random Horizontal Flipping
- Random Rotation (+- 72°)
- Random Zoom (+- 10%)
- Random Translation (+- 20%)

Vertical flipping and color augmentation were deliberately excluded as they could distort the natural appearance of hand gestures in the images of the dataset.

## CNN Architectures
Three CNN architectures of increasing complexity were designed and compared:

| Model           | Description                         | Key features                                                       |
|-----------------|-------------------------------------|--------------------------------------------------------------------|
| Model 1         | Baseline CNN                        | 2 convolutional blocks and no dropout                              |
| Model 2         | Intermediate CNN                    | 3 convolutional blocks, dropout 0.5                                |
| Model 3         | Advanced CNN                        | 4 convolutional blocks, BatchNormalization, GlobalAveragePooling2D |
| Model 2 (Tuned) | Best hyperparameters from nested CV | Model 2 retrained (train + val)                                    | 

## Methodology
### Cross-Validation and Hyperparameter Tuning

A nested cross-validation approach was adopted to simultaneously compare architectures and tune hyperparameters without introducing optimistic bias:

- **Outer loop**: 5-fold stratified CV, providing an unbiased estimate of each model's generalization performance
- **Inner loop**: 2-fold CV (inside each outer fold), performing automated grid search to find the best hyperparameters for Model 2

The hyperparameter grid searched over learning rate, dropout rate, convolutional filter configurations, and dense units. The most frequent best configuration across all outer folds was selected as the final configuration and used to retrain Model 2 on the combined train + validation data.

## Results

The three architectures showed a clear progression in performance with increasing complexity. Model 3 (Advanced) achieved the highest test accuracy, followed closely by Model 2 (Tuned), demonstrating that proper hyperparameter optimization on a simpler architecture can closely match a more complex one at lower computational cost. The nested CV results confirmed Model 2 as the most stable architecture, while Model 3 showed higher variance across folds due to BatchNormalization instability with limited data.

## Generalization Test (Optional)

To assess real-world generalization, the models were evaluated on a personal dataset of hand gesture images taken under different conditions than the training data — different backgrounds, lighting conditions, and hand appearances. All models performed at random chance level, revealing that the models learned to rely on the green screen background present in all training images rather than the hand gesture shapes themselves. This is a classic case of dataset bias and domain shift, and highlights the importance of training on diverse data for real-world deployment.

## How to Run
1. Download the dataset from Kaggle and place the class folders inside a `data/` directory
2. Run the Python script: `python rps.py`
3. Or open the Jupyter Notebook: `jupyter notebook rps.ipynb`