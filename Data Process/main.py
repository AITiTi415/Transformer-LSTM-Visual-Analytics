# main.py
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from config import SCENES_CONFIG
from preprocess_pipeline import load_driving_data, load_eye_data, extract_and_align_window

# ================= 动态配置区域 =================
DATA_ROOT = r"D:\小论文测试数据"
OUTPUT_DIR = r"D:\小论文最终清洗数据"

# 精简后的 5 维核心特征
FEATURES = ['方向盘', '油门', '刹车', '静态注视熵SGE', '注视转移熵GTE']
TARGET_FREQ = '50ms'
# ================================================

def build_multisubject_tensor():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 自动获取 DATA_ROOT 下的所有子文件夹名称作为测试人员 ID
    subject_ids = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"[*] 发现 {len(subject_ids)} 个测试人员目录: {subject_ids}")

    raw_data_store = {sub: {env: {} for env in SCENES_CONFIG} for sub in subject_ids}
    final_subject_exports = {sub: [] for sub in subject_ids}
    final_tensor_list = []
    
    print("\n>>> 阶段 1: 加载所有人员数据并生成标准化绝对时间网格...")
    for sub in subject_ids:
        for env, scenes in SCENES_CONFIG.items():
            env_name_map = {"VR厂房": "厂房", "VR戈壁": "戈壁", "VR公路": "公路"}
            file_prefix = env_name_map.get(env, "")
            
            drive_file = os.path.join(DATA_ROOT, sub, f"{file_prefix}.csv")
            eye_file = os.path.join(DATA_ROOT, sub, f"{file_prefix}眼动.csv")
            
            if not os.path.exists(drive_file):
                continue
                
            try:
                driving_df = load_driving_data(drive_file)
                eye_df = load_eye_data(eye_file) if os.path.exists(eye_file) else None
            except Exception as e:
                print(f"  [!] {sub} - {env} 加载异常: {e}")
                continue
                
            for idx, (start_sec, end_sec) in enumerate(scenes, 1):
                aligned_df = extract_and_align_window(driving_df, eye_df, start_sec, end_sec, TARGET_FREQ)
                # 【关键修复】：缩进回到 for 循环内部，确保每个场景都能被正确保存
                if not aligned_df.empty:
                    # 将底层处理出来的英文列名，完美映射为你定义的中文 FEATURES 变量
                    aligned_df.rename(columns={
                        'Steering': '方向盘',
                        'Throttle': '油门',
                        'Brake': '刹车',
                        'SGE': '静态注视熵SGE',
                        'GTE': '注视转移熵GTE'
                    }, inplace=True)
                    
                    # 确保 5 维特征完备，缺失补 0
                    for feat in FEATURES:
                        if feat not in aligned_df.columns: 
                            aligned_df[feat] = 0.0
                            
                    raw_data_store[sub][env][idx] = aligned_df[FEATURES]

    print("\n>>> 阶段 2: 独立场景异常剔除与深度学习全员偏差比打 (Z-Score)...")
    for env, scenes in SCENES_CONFIG.items():
        for idx in range(1, len(scenes) + 1):
            
            # 取出该环境该场景下，所有成功加载数据的人员
            valid_subs_for_scene = [sub for sub in subject_ids if idx in raw_data_store[sub][env]]
            if len(valid_subs_for_scene) < 2:
                continue # 人数太少无法做群体计算
                
            scene_data = np.stack([raw_data_store[sub][env][idx].values for sub in valid_subs_for_scene]).astype(np.float64)
            num_subs, time_steps, num_feats = scene_data.shape
            
            # 确保结果存放矩阵也是高精度浮点型
            compared_result = np.zeros_like(scene_data, dtype=np.float64)
            
            for f_idx, feat_name in enumerate(FEATURES):
                feature_curves = scene_data[:, :, f_idx] 
                
                # --- A: 场景内异常剔除 ---
                # 孤立森林检测该特征波形极其异常的测试人员序列
                iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(feature_curves) 
                
                valid_mask = (predictions == 1)
                if not np.any(valid_mask): valid_mask = np.ones(num_subs, dtype=bool) # 防爆保护
                
                # --- B: 计算标准共识轨迹 ---
                valid_curves = feature_curves[valid_mask]
                scene_valid_mean = np.mean(valid_curves, axis=0)
                scene_valid_std = np.std(valid_curves, axis=0)
                scene_valid_std = np.clip(scene_valid_std, a_min=1e-5, a_max=None)
                
                # --- C: 全员动态比对分析 ---
                compared_result[:, :, f_idx] = (feature_curves - scene_valid_mean) / scene_valid_std

            # --- D: 数据分发并准备导出 ---
            for i, sub in enumerate(valid_subs_for_scene):
                df_compared = pd.DataFrame(np.round(compared_result[i], 4), columns=FEATURES)
                df_compared.insert(0, 'TimeStep_Index', range(time_steps))
                df_compared.insert(0, 'Scene', f"Scene_{idx}")
                df_compared.insert(0, 'Environment', env)
                
                final_subject_exports[sub].append(df_compared)
            
                final_tensor_list.append({
                    'data': compared_result[i]
                })

    print("\n>>> 阶段 3: 输出分析文件和统一张量数据集...")
    for sub in subject_ids:
        if final_subject_exports[sub]:
            sub_combined_df = pd.concat(final_subject_exports[sub], ignore_index=True)
            # 文件名按测试人员独立输出
            out_file = os.path.join(OUTPUT_DIR, f"{sub}_Processed_Data.csv")
            sub_combined_df.to_csv(out_file, index=False, encoding='utf-8-sig')
            print(f"  [+] 已生成独立分析文件: {out_file}")

    if final_tensor_list:
        max_time_steps = max([item['data'].shape[0] for item in final_tensor_list])
        tensor_data = np.zeros((len(final_tensor_list), max_time_steps, len(FEATURES)))
        
        for i, item in enumerate(final_tensor_list):
            actual_length = item['data'].shape[0]
            tensor_data[i, :actual_length, :] = item['data']
            
        np.save('AllData_Process.npy', tensor_data)
        print(f"\n=================================================")
        print(f"全部数据处理完毕！")
        print(f"总张量形状 (用于模型训练): {tensor_data.shape} -> [Batch, Padding_Steps, 5]")
        print(f"=================================================")

if __name__ == "__main__":
    build_multisubject_tensor()