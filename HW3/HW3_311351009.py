import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np

train_df = pd.read_csv('./data/training.csv')
test_df = pd.read_csv('./data/test_X.csv')

X = train_df.iloc[:, 1:].values
y = train_df.iloc[:, 0].values
X_test = test_df.values

# 分層抽樣：因為每個標籤都有相同的數量樣本(700個)，使用 stratify=y 確保每個標籤在訓練集和驗證集中都有相同的比例
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train
rf = RandomForestClassifier(n_estimators=30, criterion='entropy')
rf.fit(X_train_scaled, y_train)

# Valid
probs_val = rf.predict_proba(X_val_scaled)

y_val_bin = label_binarize(y_val, classes=np.unique(y_train))   # Binarize the validation labels for ROC AUC calculation
if y_val_bin.shape[1] == 1:  # Check if only one class exists after binarization
    y_val_bin = np.hstack((1 - y_val_bin, y_val_bin))

roc_auc = roc_auc_score(y_val_bin, probs_val, multi_class='ovr')
print(f'ROC AUC on Validation Set: {roc_auc}')

# Test
probs_test = rf.predict_proba(X_test_scaled)

uniform_distribution = [1 / len(rf.classes_)] * len(rf.classes_)
mse_test = []
for prob in probs_test:
    mse = mean_squared_error(prob, uniform_distribution)
    inverse_mse = 1 / mse
    mse_test.append(inverse_mse)

results_df = pd.DataFrame(mse_test, columns=['outliers'])
results_df.to_csv('./random_forest.csv', index_label='id')