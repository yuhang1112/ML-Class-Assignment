import numpy as np
import pandas as pd
import scipy.io as sio

# 使用 scipy.io.loadmat 读取 .mat 文件
data_train = sio.loadmat("data_train.mat")['data_train']
label_train = sio.loadmat("label_train.mat")['label_train'].ravel()
data_test = sio.loadmat("data_test.mat")['data_test']

# divide data by class
X1 = data_train[label_train == 1]
X2 = data_train[label_train == -1]

# compute class means
m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

# within-class scatter matrix
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# weight vector w
w = np.linalg.inv(Sw) @ (m1 - m2)

# threshold w0
y0 = 0.5 * w.T @ (m1 + m2)
w0 = -y0

print("Weight vector w =", w)
print("Bias term w0 =", w0)

# discriminant function (返回标签 1 或 -1)
def predict(x):
    return 1 if w.T @ x + w0 > 0 else -1

# predict test set
preds = np.array([predict(x) for x in data_test])
pd.DataFrame(preds, columns=["PredictedClass"]).to_csv("fisher_predictions.csv", index=False)
print("Predictions saved to fisher_predictions.csv")
