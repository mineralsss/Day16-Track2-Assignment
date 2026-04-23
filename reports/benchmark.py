

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import time
import numpy as np
import json

# 1. Load dataset and measure load time
start_load = time.time()
df = pd.read_csv('creditcard.csv')
end_load = time.time()
load_time = end_load - start_load

# 2. Prepare data
X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train LightGBM model and measure training time
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1}
start_train = time.time()
model = lgb.train(params, train_data, num_boost_round=100)
end_train = time.time()
train_time = end_train - start_train

# 4. Best iteration (if early stopping is used, otherwise num_boost_round)
best_iteration = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else 100

# 5. Predict and evaluate
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 6. Inference latency (1 row)
single_row = X_test.iloc[[0]]
start_inf = time.time()
_ = model.predict(single_row)
end_inf = time.time()
latency_1row = (end_inf - start_inf) * 1000  # ms

# 7. Inference throughput (1000 rows)
if len(X_test) >= 1000:
	batch = X_test.iloc[:1000]
else:
	batch = X_test
start_throughput = time.time()
_ = model.predict(batch)
end_throughput = time.time()
throughput_1000 = end_throughput - start_throughput  # seconds


print("Metric\tKết quả")
print(f"Thời gian load data\t{load_time:.4f} s")
print(f"Thời gian training\t{train_time:.4f} s")
print(f"Best iteration\t{best_iteration}")
print(f"AUC-ROC\t{auc:.4f}")
print(f"Accuracy\t{accuracy:.4f}")
print(f"F1-Score\t{f1:.4f}")
print(f"Precision\t{precision:.4f}")
print(f"Recall\t{recall:.4f}")
print(f"Inference latency (1 row)\t{latency_1row:.4f} ms")
print(f"Inference throughput (1000 rows)\t{throughput_1000:.4f} s")

# Export results to benchmark_result.json
results = {
	"Thoi_gian_load_data_s": load_time,
	"Thoi_gian_training_s": train_time,
	"Best_iteration": int(best_iteration),
	"AUC_ROC": auc,
	"Accuracy": accuracy,
	"F1_Score": f1,
	"Precision": precision,
	"Recall": recall,
	"Inference_latency_1row_ms": latency_1row,
	"Inference_throughput_1000rows_s": throughput_1000
}
with open("benchmark_result.json", "w") as f:
	json.dump(results, f, indent=2)