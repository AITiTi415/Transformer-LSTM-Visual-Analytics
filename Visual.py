import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="人-车-环智能评估", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #FDFBF7; color: #4A4036; font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; }
    
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
    
    .crypto-card {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 10px;
        margin-bottom: 0px; 
    }
    
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

@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("能力占比分析.csv")
    except FileNotFoundError:
        st.error(" 未找到 '能力占比分析.csv'，请确认文件路径！")
        st.stop()
        
    abilities = ["专注注意力", "全面注意力", "应急响应力", "记忆学习力", 
                 "执行力", "空间认知能力", "动作抑制", "冲动行为"]
                 
    features = df[abilities].values
    
    pca = PCA(n_components=2)
    xy_coords = pca.fit_transform(features)
    
    df['X'] = xy_coords[:, 0]
    df['Y'] = xy_coords[:, 1]
    df['群体分类'] = np.where(df['X'] > np.median(df['X']), '敏锐响应型集群', '稳健控制型集群')
    
    df['主导能力'] = df[abilities].idxmax(axis=1)
    
    # 【重点】：计算全员平均基线
    global_avg = df[abilities].mean().values.tolist()
    
    return df, abilities, global_avg

df, abilities, global_avg = load_and_process_data()


st.markdown('<div class="main-title"> 人-车-环 多模态认知评估</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title-desc">基于BiLSTM+Transformer的自监督认知画像体系</div>', unsafe_allow_html=True)


total_subjects = len(df)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box"><div class="kpi-val">{total_subjects} <span style="font-size:1.2rem; color:#8D7B68;">人</span></div><div class="kpi-label">评估总人数</div></div>
    <div class="kpi-box"><div class="kpi-val">35 <span style="font-size:1.2rem; color:#8D7B68;">个</span></div><div class="kpi-label">VR多尺度虚拟环境场景数</div></div>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:10px;'>🧭 宏观测试群体空间拓扑</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A49381; font-size:0.9rem;'>点间欧式距离越近代表操作风格越相似，悬浮可查看个体【主导能力】。</p>", unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        df, x="X", y="Y", color="群体分类", hover_name="测试人员编号",
        hover_data={"主导能力": True, "X": False, "Y": False},
        color_discrete_map={"敏锐响应型集群": "#FF6B6B", "稳健控制型集群": "#FCA311"}
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
    st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:15px;'>🎯 个体与全局基线特征对比</h3>", unsafe_allow_html=True)
    
    selected_subject = st.selectbox("请选取目标人员进行深度比对：", df["测试人员编号"].tolist())
    
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

st.markdown('<div class="crypto-card" style="margin-top: -25px;">', unsafe_allow_html=True)
st.markdown("<h3 style='text-align:left; color:#E07A5F; margin-bottom:15px;'>💡 深度诊断与分析报告</h3>", unsafe_allow_html=True)

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
    "专注注意力": "眼动追踪模态显示其具备极高的单点目标凝视定力，静视熵（SGE）表现极为稳健。",
    "全面注意力": "转移熵（GTE）指标显著占优，在复杂环境中具备敏锐的全局多目标扫视检索能力。",
    "应急响应力": "制动踏板压值瞬间爆发且极快达峰，拥有顶级的危险规避本能与快速神经反应。",
    "记忆学习力": "在跨场景的控制方差波动曲线收敛极快，体现出极佳的策略自适应与经验迁移能力。",
    "执行力": "油门深浅开度与刹车转换过渡极为果断，神经肌肉动力输出干脆利落。",
    "空间认知能力": "方向盘波动方差曲线平滑，弯道轨迹构型及车道保持能力极其优秀。",
    "动作抑制": "在强干扰和复杂状态下，未触发油门误踩，动作抑制网络给予极高评级。"
}

# ---------------- 卡片 UI 参数构建 ----------------
# [左侧卡片]
max_title = f" 核心优势：【{max_ability}】"
max_border, max_bg, max_text_color = "#F4A261", "rgba(244, 162, 97, 0.12)", "#D97706"
max_tag_text = f"超越 {percentile_max:.1f}% 群体"

# [中间卡片]
if pos_scores[min_pos_idx] < (baseline_for_min * 0.85):
    min_title = f" 急需提升：【{min_ability}】"
    min_border, min_bg, min_text_color = "#E07A5F", "rgba(224, 122, 95, 0.12)", "#E07A5F"
else:
    min_title = f" 潜在短板：【{min_ability}】"
    min_border, min_bg, min_text_color = "#8D7B68", "rgba(141, 123, 104, 0.1)", "#5D4037"
min_tag_text = f"超越 {percentile_min:.1f}% 群体"

# [右侧卡片 - 动态红绿灯]
# 设定阈值：如果高于全员平均线，判定为高危；否则判定为稳定
if impulse_score > impulse_avg:
    imp_title = f" 高危预警：【{impulse_ability}】"
    imp_border, imp_bg, imp_text_color = "#E63946", "rgba(230, 57, 70, 0.12)", "#E63946"
    imp_tag_text = f"风险高于 {percentile_impulse:.1f}% 群体"
    imp_desc = "在车辆状态未完全回正时存在频繁大开度踩油门行为，属于典型的激进冒险型驾驶特征，极易引发失控。"
else:
    imp_title = f" 情绪克制极佳：【无冲动风险】"
    imp_border, imp_bg, imp_text_color = "#10B981", "rgba(16, 185, 129, 0.12)", "#059669"
    imp_tag_text = f"表现优于 {100 - percentile_impulse:.1f}% 群体"
    imp_desc = "驾驶动作克制，未检测到明显的提前抢跑或暴力踩踏油门行为，具备极佳的自我情绪约束能力。"

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
            绝对评估占比达 <span style="font-size:1.05rem; font-weight:bold; color:{max_text_color};">{scores[max_pos_idx]:.2f}%</span>。<br>{ability_badges[max_ability]}
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
            绝对评估占比为 <span style="font-size:1.05rem; font-weight:bold; color:{min_text_color};">{scores[min_pos_idx]:.2f}%</span>。<br>{ability_badges[min_ability]}
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
            冲动行为占比 <span style="font-size:1.05rem; font-weight:bold; color:{imp_text_color};">{impulse_score:.2f}%</span>。<br>{imp_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. 全局数据面板与一键导出 
# ==========================================
st.markdown("<hr style='border: 1px dashed rgba(224, 122, 95, 0.3); margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)

with st.expander(" 展开查看/导出底座原始解析数据"):
    st.markdown("<p style='color:#8D7B68; font-size:0.9rem;'>下方展示了所有人员的 8 维能力评估精准占比明细表，支持点击表头进行排序分析。</p>", unsafe_allow_html=True)
    display_df = df.drop(columns=['X', 'Y', '群体分类', '主导能力']).copy()
    for col in abilities:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
    st.dataframe(display_df, use_container_width=True)
    
    csv = df.drop(columns=['X', 'Y', '群体分类', '主导能力']).to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label=" 一键导出 CSV 分析报告",
        data=csv,
        file_name='认知能力完整报告.csv',
        mime='text/csv',
    )