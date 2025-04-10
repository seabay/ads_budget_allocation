
import pandas as pd
from quantile_model import train_quantile_model
import xgboost as xgb

# 加载数据（你可以根据需要调整）
df = pd.read_csv("your_dataset.csv")

features = ["platform", "geo", "quarter", "budget", "prev_roi", "impressions", "clicks"]
target = "return"  # 建议预测 return，再除以 budget 得到 ROI

X = df[features]
y = df[target]
budget = df["budget"].values

# 训练三个 quantile 模型
model_p10 = train_quantile_model(X, y, alpha=0.1)
model_p50 = train_quantile_model(X, y, alpha=0.5)
model_p90 = train_quantile_model(X, y, alpha=0.9)

# 推断
dtest = xgb.DMatrix(X)
return_p10 = model_p10.predict(dtest)
return_p50 = model_p50.predict(dtest)
return_p90 = model_p90.predict(dtest)

# 计算 ROI（注意除以预算）
roi_p10 = return_p10 / budget
roi_p50 = return_p50 / budget
roi_p90 = return_p90 / budget

# 存储结果
df["roi_p10"] = roi_p10
df["roi_p50"] = roi_p50
df["roi_p90"] = roi_p90
df.to_csv("roi_with_uncertainty.csv", index=False)
