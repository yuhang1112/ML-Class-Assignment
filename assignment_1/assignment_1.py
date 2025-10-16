import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

TRAIN_PATH = "TrainingData.xlsx"
TEST_PATH  = "TestData.xlsx"

train = pd.read_excel(TRAIN_PATH, header=None)
test  = pd.read_excel(TEST_PATH, header=None)

train.columns = list("ABCDE")  # A-D features, E label
test.columns  = list("ABCD")

# Replace "?" with NaN
train = train.replace("?", np.nan)
test  = test.replace("?", np.nan)

# Remove rows with missing values
train = train.dropna()
test  = test.dropna()

X_train = train[["A", "B", "C", "D"]].astype(float)
y_train = train["E"].astype(int)
X_test  = test[["A", "B", "C", "D"]].astype(float)

# Calculate class prior probabilities
classes = np.unique(y_train)
priors = {c: np.mean(y_train == c) for c in classes}

# Calculate mean and variance for each feature per class (Gaussian assumption)
mu = {c: X_train[y_train == c].mean().values for c in classes}
var = {c: X_train[y_train == c].var(ddof=0).values + 1e-9 for c in classes}

# Define log Gaussian discriminant function
def log_gaussian(x, mean, var):
    return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

def discriminant_scores(x):
    scores = {}
    for c in classes:
        scores[c] = np.log(priors[c]) + log_gaussian(x, mu[c], var[c])
    return scores

# Predict on the test set
preds, scores_list = [], []
for i, row in X_test.iterrows():
    x = row.values
    scores = discriminant_scores(x)
    scores_list.append({'id': i, **scores})
    preds.append((i, max(scores, key=scores.get)))

pred_df = pd.DataFrame(preds, columns=["SampleIndex", "PredictedClass"])
pred_df.to_csv("predictions.csv", index=False, encoding="utf-8-sig")

scores_df = pd.DataFrame(scores_list)
scores_df.to_csv("discriminant_scores.csv", index=False, encoding="utf-8-sig")

print("Finished! Predictions saved to predictions.csv")
