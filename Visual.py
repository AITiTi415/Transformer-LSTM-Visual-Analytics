import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# ==========================================
# 1. 页面全局配置 (开启宽屏模式)
# ==========================================
st.set_page_config(
    page_title="人-车-环智能评估", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. 暖色调晨曦主题 UI 样式与【悬浮动效】注入
# ==========================================
st.markdown("""
<style>
    /* 全局背景色与字体：象牙白与暖灰 */
    .stApp { background-color: #FDFBF7; color: #4A4036; font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; }
    
    /* 标题暖色渐变 */
    .main-title {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53, #FCA311);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 2px;
        padding-top: 10px;
        margin-bottom: 5px;
    }
    .sub-title-desc {
        text-align: center; color: #8D7B68; font-size: 1.05rem; margin-bottom: 25px;
    }
    
    /* 透明无界卡片 */
    .crypto-card {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 10px;
        margin-bottom: 0px; 
    }
    
    /* KPI 卡片优化：加入 Hover 平滑上浮动效 */
    .kpi-container { display: flex; justify-content: center; gap: 40px; margin-bottom: 25px; }
    .kpi-box {
        width: 300px; 
        background: rgba(255, 255, 255, 0.6); 
        border: 1.5px solid rgba(224, 122, 95, 0.4); 
        border-radius: 12px;
        padding: 15px; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(224, 122, 95, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(224, 122, 95, 0.15);
        border-color: rgba(255, 107, 107, 0.7);
    }
    .kpi-val { font-size: 2.2rem; font-weight: bold; font-family: monospace; color: #FF6B6B; }
    .kpi-label { font-size: 1rem; color: #8D7B68; margin-top: 5px; font-weight: 600;}

    /* 诊断报告卡片 Hover 动效 */
    .report-card {
        border-radius: 12px; padding: 18px; 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .report-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(141, 123, 104, 0.12);
    }

    /* 下拉选择框样式 */
    .stSelectbox label { color: #E07A5F !important; font-weight: bold; font-size: 1rem; }
    div[data-baseweb="select"] > div { 
        background-color: transparent !important; 
        border: 1px solid rgba(244, 162, 97, 0.5) !important; 
        color: #4A4036 !important;
        border-radius: 8px !important;
        transition: box-shadow 0.3s ease;
    }
    div[data-baseweb="select"] > div:hover {
        box-shadow: 0 0 10px rgba(244, 162, 97, 0.2) !important;
    }
    
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 数据处理核心 (包含全局特征计算)
# ==========================================
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("能力占比分析.csv")
    except FileNotFoundError:
        st.error("❌ 未找到 '能力占比分析.csv'，请确认文件路径！")
        st.stop()
        
    abilities = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                 "执行力", "空间认知能力", "动作抑制", "冲动行为"]
                 
    features = df[abilities].values
    
    pca = PCA(n_components=2)
    xy_coords = pca.fit_transform(features)
    
    df['X'] = xy_coords[:, 0]
    df['Y'] = xy_coords[:, 1]
    df['Group classification'] = np.where(df['X'] > np.median(df['X']), 'Sensitive Response Cluster', 'Steady Control Type Cluster')
    
    df['主导能力'] = df[abilities].idxmax(axis=1)
    
    # 【重点】：计算全员平均基线
    global_avg = df[abilities].mean().values.tolist()
    
    return df, abilities, global_avg

df, abilities, global_avg = load_and_process_data()

# ==========================================
# 4. 顶导渲染
# ==========================================
st.markdown('<div class="main-title">🌅 Human-Vehicle-Loop Multimodal Cognitive Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title-desc">Self-supervised cognitive profile system based on BiLSTM + Transformer</div>', unsafe_allow_html=True)

# ==========================================
# 5. 精简版 KPI 数据带 
# ==========================================
total_subjects = len(df)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box"><div class="kpi-val">{total_subjects} <span style="font-size:1.2rem; color:#8D7B68;">People</span></div><div class="kpi-label">Total number of participants evaluated</div></div>
    <div class="kpi-box"><div class="kpi-val">35 <span style="font-size:1.2rem; color:#8D7B68;">Kinds</span></div><div class="kpi-label">The number of VR multi-scale virtual environment scenes</div></div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. 两栏式排版
# ==========================================
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:10px;'>🧭 Macroscopic testing of group spatial topology</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A49381; font-size:0.9rem;'>The closer the Euclidean distance between points indicates that the operation styles are more similar. You can view the individual's dominant ability by hovering.</p>", unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        df, x="X", y="Y", color="Group classification", hover_name="测试人员编号",
        hover_data={"主导能力": True, "X": False, "Y": False},
        color_discrete_map={"Sensitive Response Cluster": "#FF6B6B", "Steady Control Type Cluster": "#FCA311"}
    )
    
    fig_scatter.update_traces(
        marker=dict(size=14, opacity=0.85, line=dict(width=1.5, color='white'), symbol='circle'),
        selector=dict(mode='markers')
    )
    
    fig_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(141, 123, 104, 0.15)', zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(141, 123, 104, 0.15)', zeroline=False, showticklabels=False, title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#4A4036")),
        height=480,
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=14, font_family="Microsoft YaHei")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:15px;'>🎯 Comparison of individual and global baseline characteristics</h3>", unsafe_allow_html=True)
    
    selected_subject = st.selectbox("Please select the target individuals for a detailed comparison:", df["测试人员编号"].tolist())
    
    person_data = df[df["测试人员编号"] == selected_subject].iloc[0]
    scores = person_data[abilities].values.tolist()
    
    fig_radar = go.Figure()
    
    # 轨迹 1：全局平均基线（虚线）
    fig_radar.add_trace(go.Scatterpolar(
        r=global_avg + [global_avg[0]],
        theta=abilities + [abilities[0]],
        fill='none',
        line=dict(color='rgba(141, 123, 104, 0.6)', width=2, dash='dash'),
        name='全员平均基线',
        hoverinfo='skip'
    ))
    
    # 轨迹 2：个体真实得分（实线）
    fig_radar.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  
        theta=abilities + [abilities[0]],
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.2)',
        line=dict(color='#FF6B6B', width=3, dash='solid'),
        marker=dict(size=7, color='#FFFFFF', line=dict(width=2, color='#FF6B6B')),
        name=selected_subject
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, max(max(scores), max(global_avg)) * 1.15], 
                showticklabels=True, 
                tickangle=0,
                tickfont=dict(size=10, color="#A49381"),
                gridcolor='rgba(224, 122, 95, 0.15)',
                linecolor='rgba(224, 122, 95, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(224, 122, 95, 0.15)',
                linecolor='rgba(224, 122, 95, 0.2)',
                tickfont=dict(color="#4A4036", size=13, family="Microsoft YaHei", weight="bold")
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=420,
        margin=dict(l=100, r=100, t=80, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. AI 智能化诊断报告卡 (三栏解耦重构版)
# ==========================================
st.markdown('<div class="crypto-card" style="margin-top: -25px;">', unsafe_allow_html=True)
st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:15px;'>💡 In-depth Diagnosis and Analysis Report</h3>", unsafe_allow_html=True)

# ---------------- 核心逻辑拆解：隔离正向指标与负向指标 ----------------
# 前 7 项为正向能力，第 8 项为负向冲动行为
pos_abilities = abilities[:7]
pos_scores = scores[:7]
impulse_ability = abilities[7]
impulse_score = scores[7]
impulse_avg = global_avg[7]

# 1. 计算左侧：最强正向优势
max_pos_idx = np.argmax(pos_scores)
max_ability = pos_abilities[max_pos_idx]
percentile_max = (df[max_ability] <= pos_scores[max_pos_idx]).mean() * 100

# 2. 计算中间：最弱正向能力 (潜在提升点)
min_pos_idx = np.argmin(pos_scores)
min_ability = pos_abilities[min_pos_idx]
percentile_min = (df[min_ability] <= pos_scores[min_pos_idx]).mean() * 100
baseline_for_min = global_avg[min_pos_idx]

# 3. 计算右侧：冲动行为独立监控
percentile_impulse = (df[impulse_ability] <= impulse_score).mean() * 100

ability_badges = {
    "专注注意力": "The eye-tracking modality demonstrates an extremely high single-point target fixation stability, and the static visual entropy (SGE) performance is extremely robust.",
    "全面注意力": "The transfer entropy (GTE) index is significantly superior and possesses a keen ability to conduct global multi-objective scanning and retrieval in complex environments.",
    "应急响应力": "The braking pedal pressure value surges instantly and reaches its peak extremely quickly, demonstrating top-notch instinctive risk avoidance and rapid neural response.",
    "记忆学习力": "The variance fluctuation curves of the cross-scenario control converge extremely quickly, demonstrating excellent adaptive strategies and experience transfer capabilities.",
    "执行力": "The throttle opening and braking transition were extremely decisive, and the nerve-muscle power output was crisp and precise.",
    "空间认知能力": "The steering wheel fluctuation variance curve is smooth, and the trajectory configuration of the curves and the lane keeping ability are extremely excellent.",
    "动作抑制": "Under strong interference and complex conditions, without triggering the throttle misstep, the action suppression network received an extremely high rating."
}

# ---------------- 卡片 UI 参数构建 ----------------
# [左侧卡片]
max_title = f"🌟 核心优势：【{max_ability}】"
max_border, max_bg, max_text_color = "#F4A261", "rgba(244, 162, 97, 0.12)", "#D97706"
max_tag_text = f"Exceeding {percentile_max:.1f}% group"

# [中间卡片]
if pos_scores[min_pos_idx] < (baseline_for_min * 0.85):
    min_title = f"⚠️ Urgent need for improvement：【{min_ability}】"
    min_border, min_bg, min_text_color = "#E07A5F", "rgba(224, 122, 95, 0.12)", "#E07A5F"
else:
    min_title = f"📝 Potential weakness：【{min_ability}】"
    min_border, min_bg, min_text_color = "#8D7B68", "rgba(141, 123, 104, 0.1)", "#5D4037"
min_tag_text = f"Exceeding {percentile_min:.1f}% group"

# [右侧卡片 - 动态红绿灯]
# 设定阈值：如果高于全员平均线，判定为高危；否则判定为稳定
if impulse_score > impulse_avg:
    imp_title = f"🚨 High-risk warning：【{impulse_ability}】"
    imp_border, imp_bg, imp_text_color = "#E63946", "rgba(230, 57, 70, 0.12)", "#E63946"
    imp_tag_text = f"Groups with a risk higher than {percentile_impulse:.1f}% "
    imp_desc = "When the vehicle is not in a fully stable state, there is frequent heavy throttle application with large opening degrees, which is a typical characteristic of aggressive and risky driving behavior and is highly likely to cause loss of control."
else:
    imp_title = f"🛡️ Emotionally well-controlled: [No risk of impulsiveness]"
    imp_border, imp_bg, imp_text_color = "#10B981", "rgba(16, 185, 129, 0.12)", "#059669"
    imp_tag_text = f"Outperforming {100 - percentile_impulse:.1f}% group"
    imp_desc = "The driving actions were well-controlled, no obvious premature acceleration or violent throttle pressing was detected, and the driver demonstrated excellent self-emotional restraint ability."

# ---------------- 三栏渲染 ----------------
col_rep1, col_rep2, col_rep3 = st.columns(3)

with col_rep1:
    st.markdown(f"""
    <div class="report-card" style="background: {max_bg}; border-left: 5px solid {max_border}; height: 180px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{max_text_color}; font-weight:bold; font-size:1.1rem;">{max_title}</span>
            <span style="background:{max_border}; color:white; padding:3px 8px; border-radius:15px; font-size:0.75rem; font-weight:bold;">{max_tag_text}</span>
        </div>
        <p style="margin-top:12px; color:#5D4037; font-size:0.9rem; line-height:1.6;">
            The proportion of absolute assessment is <span style="font-size:1.05rem; font-weight:bold; color:{max_text_color};">{scores[max_pos_idx]:.2f}%</span>。<br>{ability_badges[max_ability]}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_rep2:
    st.markdown(f"""
    <div class="report-card" style="background: {min_bg}; border-left: 5px solid {min_border}; height: 180px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{min_text_color}; font-weight:bold; font-size:1.1rem;">{min_title}</span>
            <span style="background:{min_border}; color:white; padding:3px 8px; border-radius:15px; font-size:0.75rem; font-weight:bold;">{min_tag_text}</span>
        </div>
        <p style="margin-top:12px; color:#5D4037; font-size:0.9rem; line-height:1.6;">
            The proportion of absolute assessment is <span style="font-size:1.05rem; font-weight:bold; color:{min_text_color};">{scores[min_pos_idx]:.2f}%</span>。<br>{ability_badges[min_ability]}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_rep3:
    st.markdown(f"""
    <div class="report-card" style="background: {imp_bg}; border-left: 5px solid {imp_border}; height: 180px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{imp_text_color}; font-weight:bold; font-size:1.1rem;">{imp_title}</span>
            <span style="background:{imp_border}; color:white; padding:3px 8px; border-radius:15px; font-size:0.75rem; font-weight:bold;">{imp_tag_text}</span>
        </div>
        <p style="margin-top:12px; color:#5D4037; font-size:0.9rem; line-height:1.6;">
            The proportion of impulsive behavior is <span style="font-size:1.05rem; font-weight:bold; color:{imp_text_color};">{impulse_score:.2f}%</span>。<br>{imp_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. 全局数据面板与一键导出 
# ==========================================
st.markdown("<hr style='border: 1px dashed rgba(224, 122, 95, 0.3); margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)

with st.expander("📁 Expand to view / Export the original parsing data of the base"):
    st.markdown("<p style='color:#8D7B68; font-size:0.9rem;'>Below is a detailed table showing the precise percentage breakdown of the 8-dimensional capability assessment for all personnel, and it supports sorting and analysis by clicking on the table headers.</p>", unsafe_allow_html=True)
    display_df = df.drop(columns=['X', 'Y', 'Group classification', '主导能力']).copy()
    for col in abilities:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
    st.dataframe(display_df, use_container_width=True)
    
    csv = df.drop(columns=['X', 'Y', 'Group classification', '主导能力']).to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 One-click export of CSV analysis report",
        data=csv,
        file_name='认知能力完整报告.csv',
        mime='text/csv',
    )