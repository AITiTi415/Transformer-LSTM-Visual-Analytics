import pandas as pd
import numpy as np
from collections import Counter
import math

def load_driving_data(file_path):
    """极简版：完全无视 Timestamp，直接用 Timediff 作为绝对时间轴"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    df.set_index(pd.to_timedelta(df['Timediff'], unit='ms'), inplace=True)
    df.sort_index(inplace=True)
    
    if 'Throttle' in df.columns:
        df['Throttle'] = df['Throttle'].astype(str).str.replace('%', '').astype(float)
    if 'Brake' in df.columns:
        df['Brake'] = df['Brake'].astype(str).str.replace('%', '').astype(float)
        
    return df[['Steering', 'Throttle', 'Brake']]

def calc_shannon_entropy(series):
    """计算香农熵 (SGE)"""
    counts = series.value_counts()
    probs = counts / len(series)
    return -np.sum(probs * np.log2(probs + 1e-9))

def calc_transition_entropy(series):
    """计算马尔可夫注视转移熵 (GTE)"""
    transitions = list(zip(series[:-1], series[1:]))
    if not transitions: return 0.0
    trans_counts = Counter(transitions)
    state_counts = Counter(series[:-1])
    gte = 0.0
    for (state_i, state_j), count in trans_counts.items():
        p_i = state_counts[state_i] / len(series[:-1])
        p_j_given_i = count / state_counts[state_i]
        gte -= p_i * (p_j_given_i * math.log2(p_j_given_i + 1e-9))
    return gte

def load_eye_data(file_path):
    """极简版：同样只认 timediff 作为绝对时间轴"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    df.set_index(pd.to_timedelta(df['timediff'], unit='ms'), inplace=True)
    df.sort_index(inplace=True)
    # 坐标网格化
    grid_size = 10
    df['grid_x'] = (df['x'] * grid_size).clip(0, grid_size - 1).astype(int)
    df['grid_y'] = (df['y'] * grid_size).clip(0, grid_size - 1).astype(int)
    df['grid_id'] = df['grid_x'] * grid_size + df['grid_y']
    
    # 滑动窗口计算熵值
    window_samples = max(1, len(df) // 20) if len(df) > 20 else 2 
    df['SGE'] = df['grid_id'].rolling(window=window_samples, min_periods=2).apply(calc_shannon_entropy)
    
    gte_list = [0.0] * len(df)
    grid_ids = df['grid_id'].values
    for i in range(window_samples, len(df)):
        window_data = grid_ids[i-window_samples : i]
        gte_list[i] = calc_transition_entropy(window_data)
    df['GTE'] = gte_list
    
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df[['SGE', 'GTE']]

def extract_and_align_window(driving_df, eye_df, start_sec, end_sec, target_freq='50ms'):
    start_td = pd.to_timedelta(start_sec, unit='s')
    end_td = pd.to_timedelta(end_sec, unit='s')
    
    # 构建绝对时间网格
    target_index = pd.timedelta_range(start=start_td, end=end_td, freq=target_freq)
    concat_list = []
    
    def align_to_grid(df):
        if df is None or df.empty:
            return pd.DataFrame(index=target_index)
            
        # 截取稍微大一点的窗口，保证插值时头尾有数据支撑
        pad_time = pd.Timedelta(seconds=1)
        win = df.loc[start_td - pad_time : end_td + pad_time]
        if win.empty:
            return pd.DataFrame(index=target_index)
            
        # 1. 把绝对网格“强行塞入”原数据的时间轴里
        combined = pd.concat([win, pd.DataFrame(index=target_index)])
        combined = combined.sort_index()
        # 去除碰巧时间完全一样导致的重复行
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # 2. 按时间真实距离进行线性插值
        interpolated = combined.interpolate(method='time')
        
        # 3. 抽身而出：只保留我们在绝对网格上的数据！
        aligned = interpolated.reindex(target_index)
        return aligned

    drive_aligned = align_to_grid(driving_df)
    concat_list.append(drive_aligned)
    
    if eye_df is not None:
        eye_aligned = align_to_grid(eye_df)
        concat_list.append(eye_aligned)
        
    aligned_window_df = pd.concat(concat_list, axis=1)
    
    # 填充边界可能残留的空值
    aligned_window_df = aligned_window_df.bfill().ffill().fillna(0.0)
    
    return aligned_window_df