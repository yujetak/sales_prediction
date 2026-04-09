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
    model_path = "./model/model.keras"
    encoder_path = "./model/oe.pkl"
    scaler_path = "./model/ss.pkl"
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

@st.cache_resource
def get_sorted_lists(sources):
    sorted_sources = sorted(sources.items(), key=lambda item: item[1])
    names = [item[1] for item in sorted_sources]
    codes = [item[0] for item in sorted_sources]
    return names, codes

with st.spinner("🚀 서비스 작업을 준비하는 중입니다..."):
    model, encoder, scaler, industry_type_mapping, region_mapping = load_assets()
    df = load_dataset()

if model is None:
    st.error("필수 파일(모델, 인코더, 매핑 파일 등)을 찾을 수 없습니다.")
else:
    known_regions = encoder.categories_[0].tolist()
    known_industries = encoder.categories_[2].tolist()

    _, industry_codes = get_sorted_lists(industry_type_mapping)
    _, region_codes = get_sorted_lists(region_mapping)

    st.sidebar.header("조건 입력")
    
    valid_region_codes = [c for c in region_codes if int(c) in known_regions]
    region_code = st.sidebar.selectbox(
        "📌 경기도 행정구역 선택", 
        options=valid_region_codes, 
        format_func=lambda x: f"{region_mapping[x]} ({x})"
    )
    
    valid_industry_codes = [c for c in industry_codes if str(c) in known_industries]
    industry_code = st.sidebar.selectbox(
        "🏢 업종 선택", 
        options=valid_industry_codes,
        format_func=lambda x: f"{industry_type_mapping[x]} ({x})"
    )

    time_zone = st.sidebar.multiselect(
        "🕰️ 영업 시간대 선택", 
        ["dawn", "morning", "afternoon", "night"],
        format_func=lambda x: {
            "dawn": "새벽 (00:00 ~ 06:00)", 
            "morning": "아침 (06:00 ~ 13:00)",
            "afternoon": "오후 (13:00 ~ 20:00)", 
            "night": "밤 (20:00 ~ 24:00)"
        }[x]
    )
    
    if st.button("매출 예측하기", use_container_width=True):
        if not time_zone:
            st.warning("시간대를 하나 이상 선택해주세요!")
        else:
            predicted_results = []
            progress_bar = st.progress(0)
            total_sales_sum = 0
            
            target_year, target_month = 2025, 7
            target_sin = np.sin(2 * np.pi * target_month/12.0)
            target_cos = np.cos(2 * np.pi * target_month/12.0)
            
            TIME_STEPS = 12
            feature_cols = ['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group', 'year', 'month_sin', 'month_cos', 'sales_amt_log']
            oe_cols = ['signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            ss_cols = ['year', 'signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            
            for idx, tz in enumerate(time_zone):
                # 2. 실제 과거 12개월 데이터 추출
                cond = (df['signgu_cd'] == int(region_code)) & \
                       (df['mdclass_indutype_cd'] == industry_code) & \
                       (df['tmzon_group'] == tz)
                
                recent_data = df[cond].sort_values('std_ym').tail(TIME_STEPS)
                
                if len(recent_data) < TIME_STEPS:
                    st.warning(f"'{tz}' 시간대는 과거 데이터가 부족하여 예측이 정확하지 않을 수 있습니다. (현재 {len(recent_data)}개월분)")
                    # 데이터가 부족하면 부족한 만큼 평균값 등으로 채우거나 건너뛰는 로직이 필요할 수 있습니다.
                    continue

                # 인코딩 및 스케일링 준비를 위한 복사
                input_df = recent_data.copy()
                
                # Transform (인코더와 스케일러 적용)
                input_df[oe_cols] = encoder.transform(input_df[oe_cols])
                input_df[ss_cols] = scaler.transform(input_df[ss_cols])
                
                # 모델 입력 형태 변환 (1, 12, 7)
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