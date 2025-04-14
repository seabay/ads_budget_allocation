
import streamlit as st
import pandas as pd
import joblib
from allocator import QuarterBudgetAllocator

# --- 页面设置 ---
st.set_page_config(page_title="RL 广告预算分配器", layout="wide")
st.title("🚀 广告预算分配 Demo（CQL + RL）")

# --- 侧边栏参数输入 ---
st.sidebar.header("参数设置")

total_budget = st.sidebar.number_input("季度总预算", min_value=100000, value=1000000, step=50000)

platform_max_ratio = {}
st.sidebar.markdown("### 平台最大预算比例")
for pid in range(3):
    platform_max_ratio[pid] = st.sidebar.slider(f"Platform {pid} Max Ratio", 0.1, 1.0, 0.4, 0.05)

geo_min_budget = {}
st.sidebar.markdown("### Geo 最小预算")
for gid in [0, 5, 8]:
    geo_min_budget[gid] = st.sidebar.number_input(f"Geo {gid} Min Budget", min_value=0, value=100000, step=10000)

# --- 上传 CSV 或使用示例数据 ---
st.markdown("### 📤 上传 Segment 状态 CSV")
uploaded_file = st.file_uploader("包含 roi_p10/50/90, cvr, ctr, prev_spend, platform_id, geo_id 等列", type="csv")

if uploaded_file is not None:
    segment_df = pd.read_csv(uploaded_file)
else:
    st.info("使用示例数据进行测试")
    segment_df = pd.read_csv("sample/segment_state_sample.csv")

# --- 初始化 RL 分配器 ---
scaler = joblib.load("utils/scaler.pkl")
allocator = QuarterBudgetAllocator("models/cql_ads_budget_model.pt", scaler)

# --- 分配预算 ---
if st.button("🎯 开始分配"):
    result_df = allocator.allocate(
        segment_df,
        total_budget=total_budget,
        platform_max_ratio=platform_max_ratio,
        geo_min_budget=geo_min_budget
    )
    st.success("✅ 预算分配完成")

    # 显示结果
    st.dataframe(result_df[['platform_id', 'geo_id', 'allocated_budget', 'roi']])

    # 模拟收益
    reward = allocator.simulate_reward(result_df, roi_column='roi')
    st.metric("🧮 模拟总收益（allocated_budget × roi）", f"{reward:,.2f}")
