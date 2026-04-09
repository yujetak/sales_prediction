import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json

st.set_page_config(page_icon="💰", page_title="카드매출 예측", layout="wide")

st.title("지역, 시간대, 업종의 다음달 카드매출 예측 (BiLSTM)")
st.write("경기도의 행정구역, 매장의 영업시간대, 업종을 선택하면 다음달의 카드 매출을 예측합니다")

@st.cache_resource
def load_assets():
    model_path = "./models/model.keras"
    encoder_path = "./models/oe.pkl"
    scaler_path = "./models/ss.pkl"
    industry_type_path = "./sources/industry_type_mapping.json"
    region_path = "./sources/region_mapping.json"
    
    if not all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path, industry_type_path, region_path]):
        return None, None, None, None, None
    
    model = tf.keras.models.load_model(model_path, compile=False)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    with open(industry_type_path, "r", encoding="utf-8") as f:
        industry_type_mapping = json.load(f)
        
    with open(region_path, "r", encoding="utf-8") as f:
        region_mapping = json.load(f)
  
    return model, encoder, scaler, industry_type_mapping, region_mapping

@st.cache_data
def load_dataset():
    path = './dataset/card_sales_summary_small.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)

        df = df[df['tmzon_cd'] != 'TOT'].copy()
        df['sales_amt_log'] = np.log1p(df['sales_amt'])
        df['year'] = df['std_ym'] // 100
        df['month'] = df['std_ym'] % 100
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
        
        def compute_tmzone_group(tz_cd):
            tm = int(tz_cd[-2:])
            if tm > 8: return 'night'
            elif tm > 5: return 'afternoon'
            elif tm > 2: return 'morning'
            else: return 'dawn'
        
        if 'tmzon_group' not in df.columns:
            df['tmzon_group'] = df['tmzon_cd'].apply(compute_tmzone_group)
            
        return df
    return None

with st.spinner("🚀 서비스 작업을 준비하는 중입니다..."):
    model, encoder, scaler, industry_type_mapping, region_mapping = load_assets()
    df = load_dataset()

if model is None or df is None:
    st.error("필수 파일(모델, 데이터셋 등)을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    counts = df.groupby(['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group']).size().reset_index(name='count')
    valid_combos = counts[counts['count'] >= 12].copy()
    
    st.sidebar.header("조건 입력")
    
    valid_region_codes = sorted(valid_combos['signgu_cd'].unique().astype(str).tolist())
    
    region_code = st.sidebar.selectbox(
        "📌 경기도 행정구역 선택", 
        options=valid_region_codes, 
        format_func=lambda x: f"{region_mapping.get(x, x)} ({x})"
    )
    
    valid_industry_codes = sorted(valid_combos[
        valid_combos['signgu_cd'] == int(region_code)
    ]['mdclass_indutype_cd'].unique().tolist())

    industry_code = st.sidebar.selectbox(
        "🏢 업종 선택", 
        options=valid_industry_codes,
        format_func=lambda x: f"{industry_type_mapping.get(x, x)} ({x})"
    )

    valid_tz_options = valid_combos[
        (valid_combos['signgu_cd'] == int(region_code)) & 
        (valid_combos['mdclass_indutype_cd'] == industry_code)
    ]['tmzon_group'].unique().tolist()

    time_zone = st.sidebar.multiselect(
        "🕰️ 영업 시간대 선택", 
        options=valid_tz_options,
        format_func=lambda x: {
            "dawn": "새벽 (00:00 ~ 06:00)", 
            "morning": "아침 (06:00 ~ 13:00)",
            "afternoon": "오후 (13:00 ~ 20:00)", 
            "night": "밤 (20:00 ~ 24:00)"
        }.get(x, x)
    )
    
    if st.button("매출 예측하기", use_container_width=True):
        if not time_zone:
            st.warning("시간대를 하나 이상 선택해주세요!")
        else:
            predicted_results = []
            progress_bar = st.progress(0)
            total_sales_sum = 0
            
            TIME_STEPS = 12
            feature_cols = ['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group', 'year', 'month_sin', 'month_cos', 'sales_amt_log']
            oe_cols = ['signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            ss_cols = ['year', 'signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            
            for idx, tz in enumerate(time_zone):
                cond = (df['signgu_cd'] == int(region_code)) & \
                       (df['mdclass_indutype_cd'] == industry_code) & \
                       (df['tmzon_group'] == tz)
                
                recent_data = df[cond].sort_values('std_ym').tail(TIME_STEPS)
                
                input_df = recent_data.copy()
                
                # Transform
                input_df[oe_cols] = encoder.transform(input_df[oe_cols])
                input_df[ss_cols] = scaler.transform(input_df[ss_cols])
                
                model_input = input_df[feature_cols].values.reshape(1, TIME_STEPS, len(feature_cols))
                
                prediction = model.predict(model_input, verbose=0)
                result_val = np.expm1(prediction)[0][0]
                
                total_sales_sum += result_val
                predicted_results.append({"시간대": tz, "예상 매출액": f"{int(result_val):,} 원"})
                progress_bar.progress((idx + 1) / len(time_zone))
            
            progress_bar.empty()
            st.divider()
            st.metric(label="🎶 총 예상 매출액", value=f"{int(total_sales_sum):,} 원")
            st.subheader("🎯 시간대별 상세 결과")
            st.table(pd.DataFrame(predicted_results))