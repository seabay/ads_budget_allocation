
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("roi_with_uncertainty.csv")

# 示例：按 platform + geo 画 ROI 置信区间
for geo in df["geo"].unique():
    subset = df[df["geo"] == geo]
    plt.errorbar(subset["platform"], subset["roi_p50"],
                 yerr=[subset["roi_p50"] - subset["roi_p10"], subset["roi_p90"] - subset["roi_p50"]],
                 fmt='o', label=geo)

plt.title("ROI Prediction with 80% Confidence Interval")
plt.ylabel("Predicted ROI")
plt.xlabel("Platform")
plt.legend()
plt.grid(True)
plt.show()
