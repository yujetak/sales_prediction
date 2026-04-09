import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import os
import json
import altair as alt

st.set_page_config(page_icon="💰", page_title="카드매출 예측", layout="wide")

st.markdown("""
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        white-space: normal !important;
        height: auto !important;
        min-height: 32px;
        max-width: 100% !important;
    }
    .stMultiSelect [data-baseweb="tag"] > span {
        overflow: visible !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 경기도 내 지역·업종·시간대별 카드 매출 예측 서비스")

st.markdown(f"""
**[프로젝트 배경 및 목적]**
* **자영업 생존 위기:** 고물가·고금리로 인한 내수 침체 및 경제 악화로 연간 폐업자 **100만 명** 돌파
* **정보 접근 한계:** 상권 매출 데이터는 존재하지만, 대규모 데이터셋을 직접 조회·분석하기 어려움
* **모델 구축:** 카드 빅데이터와 시계열 딥러닝 모델을 결합하여 매출 및 변동률 산출 
* **분석 지원:** 과거 1년 매출 시각화와 매출 예측을 함께 제시하여 정량 분석 지원
""")

st.divider()

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

def format_korean_currency(amount):
    amount = int(amount)
    if amount >= 100000000:
        billion = amount // 100000000
        ten_thousand = (amount % 100000000) // 10000
        if ten_thousand > 0:
            return f"{billion}억 {ten_thousand:,}만 원"
        return f"{billion}억 원"
    elif amount >= 10000:
        ten_thousand = amount // 10000
        return f"{ten_thousand:,}만 원"
    else:
        return f"{amount:,} 원"

with st.spinner("🚀 서비스 작업을 준비하는 중입니다..."):
    model, encoder, scaler, industry_type_mapping, region_mapping = load_assets()
    df = load_dataset()

if model is None or df is None:
    st.error("필수 파일들을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    counts = df.groupby(['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group'])['std_ym'].nunique().reset_index(name='count')
    valid_combos = counts[counts['count'] >= 12].copy()
    
    st.sidebar.header("조건 입력")
    st.sidebar.write("")
    
    valid_region_codes = sorted(valid_combos['signgu_cd'].unique().astype(str).tolist())
    
    region_code = st.sidebar.selectbox(
        "📌 상권(지역) 선택", 
        options=valid_region_codes, 
        format_func=lambda x: f"{region_mapping.get(x, x)} ({x})"
    )
    
    st.sidebar.divider()
    
    valid_industry_codes = sorted(valid_combos[
        valid_combos['signgu_cd'] == int(region_code)
    ]['mdclass_indutype_cd'].unique().tolist())

    industry_code = st.sidebar.selectbox(
        "🏢 업종 선택", 
        options=valid_industry_codes,
        format_func=lambda x: f"{industry_type_mapping.get(x, x)} ({x})"
    )

    st.sidebar.divider()

    valid_tz_options = valid_combos[
        (valid_combos['signgu_cd'] == int(region_code)) & 
        (valid_combos['mdclass_indutype_cd'] == industry_code)
    ]['tmzon_group'].unique().tolist()

    time_zone = st.sidebar.multiselect(
        "🕰️ 영업 시간대 선택", 
        options=valid_tz_options,
        format_func=lambda x: {
            "dawn": "새벽 (00~06시)", 
            "morning": "아침 (06~13시)",
            "afternoon": "오후 (13~20시)", 
            "night": "밤 (20~24시)"
        }.get(x, x)
    )
    
    st.sidebar.divider()
    
    if st.button("매출 예측하기", use_container_width=True):
        if not time_zone:
            st.warning("시간대를 하나 이상 선택해주세요!")
        else:
            predicted_results = []
            chart_data = pd.DataFrame()
            progress_bar = st.progress(0)
            total_sales_sum = 0
            total_last_sum = 0
            
            TIME_STEPS = 12
            feature_cols = ['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group', 'year', 'month_sin', 'month_cos', 'sales_amt_log']
            oe_cols = ['signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            ss_cols = ['year', 'signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            
            TIME_ORDER = {"dawn": 0, "morning": 1, "afternoon": 2, "night": 3}
            sorted_time_zones = sorted(time_zone, key=lambda x: TIME_ORDER.get(x, 99))
            
            for idx, tz in enumerate(sorted_time_zones):
                cond = (df['signgu_cd'] == int(region_code)) & \
                       (df['mdclass_indutype_cd'] == industry_code) & \
                       (df['tmzon_group'] == tz)
                
                grouped_data = df[cond].groupby('std_ym').agg({
                    'sales_amt': 'mean', 'year': 'first', 'month': 'first',
                    'tmzon_group': 'first', 'signgu_cd': 'first', 'tmzon_cd': 'first',
                    'mdclass_indutype_cd': 'first'
                }).reset_index()
                
                grouped_data['sales_amt_log'] = np.log1p(grouped_data['sales_amt'])
                grouped_data['month_sin'] = np.sin(2 * np.pi * grouped_data['month'] / 12.0)
                grouped_data['month_cos'] = np.cos(2 * np.pi * grouped_data['month'] / 12.0)
                
                recent_data = grouped_data.sort_values('std_ym').tail(TIME_STEPS)
                
                if len(recent_data) < TIME_STEPS:
                    st.warning(f"'{tz}' 시간대는 과거 데이터가 부족하여 예측이 정확하지 않을 수 있습니다. (현재 {len(recent_data)}개월분)")
                    continue
                
                input_df = recent_data.copy()
                input_df[oe_cols] = encoder.transform(input_df[oe_cols])
                input_df[ss_cols] = scaler.transform(input_df[ss_cols])
                
                model_input = input_df[feature_cols].values.reshape(1, TIME_STEPS, len(feature_cols))
                
                prediction = model.predict(model_input, verbose=0)
                result_val = np.expm1(prediction)[0][0]
                
                actual_sales = result_val * 1000
                total_sales_sum += actual_sales
                
                last_actual_sales = recent_data.iloc[-1]['sales_amt'] * 1000
                total_last_sum += last_actual_sales
                growth_rate = ((actual_sales - last_actual_sales) / last_actual_sales) * 100 if last_actual_sales > 0 else 0.0
                
                tz_label = {"dawn": "새벽", "morning": "아침", "afternoon": "오후", "night": "밤"}.get(tz, tz)
                
                last_std_ym = recent_data.iloc[-1]['std_ym']
                n_year = last_std_ym // 100
                n_month = last_std_ym % 100
                if n_month == 12:
                    n_year += 1
                    n_month = 1
                else:
                    n_month += 1
                next_std_ym = n_year * 100 + n_month
                
                all_ym = recent_data['std_ym'].tolist() + [next_std_ym]
                str_dates = [str(ym)[2:] for ym in all_ym]
                values = (recent_data['sales_amt'] * 1000).tolist() + [actual_sales]
                
                if chart_data.empty:
                    chart_data['결제 시기 (YYMM)'] = str_dates
                    chart_data = chart_data.set_index('결제 시기 (YYMM)')
                chart_data[tz_label] = values
                
                growth_str = f"▲ {growth_rate:.1f}%" if growth_rate > 0 else f"▼ {growth_rate:.1f}%" if growth_rate < 0 else "0.0%"
                
                # 예측 결과 수집 (스타일은 나중에 Pandas Styler로 일괄 적용)
                predicted_results.append({
                    "시간대": tz_label, 
                    "상권 월간 예상 매출": format_korean_currency(actual_sales),
                    "성장률": growth_str
                })
                progress_bar.progress((idx + 1) / len(sorted_time_zones))
            
            progress_bar.empty()
            st.divider()
            
            # 1. 종합 지표 표시 (Metric)
            total_growth = ((total_sales_sum - total_last_sum) / total_last_sum) * 100 if total_last_sum > 0 else 0.0
            district_name = region_mapping.get(str(region_code), str(region_code))
            
            st.metric(
                label=f"🎶 [{district_name} 지역 상권] 다음 달 예상 매출액", 
                value=format_korean_currency(total_sales_sum),
                delta=f"전월 대비 {total_growth:+.1f}%"
            )
            
            st.divider()
            
            # 2. 상세 결과 테이블 (Pandas Styler 활용)
            st.subheader("🎯 시간대별 상세 예측 결과")
            display_df = pd.DataFrame(predicted_results)
            
            def apply_custom_style(styler):
                # 시간대별 감성 컬러 (새벽:보라, 아침:초록, 오후:주황, 밤:파랑)
                tz_colors = {"새벽": "#9333ea", "아침": "#22c55e", "오후": "#f97316", "밤": "#2563eb"}
                styler.map(lambda v: f"color: {tz_colors.get(v, 'black')}; font-weight: bold;", subset=['시간대'])
                
                styler.map(lambda v: f"color: {'#ef4444' if '▲' in str(v) else '#3b82f6' if '▼' in str(v) else '#6b7280'}; font-weight: bold;", subset=['성장률'])
                
                styler.set_properties(**{'text-align': 'center', 'border': '1px solid #d1d5db'})
                
                styler.set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#f3f4f6'), ('color', '#111827'), 
                        ('font-weight', 'bold'), ('text-align', 'center'), 
                        ('padding', '10px'), ('border', '1px solid #d1d5db')
                    ]},
                    {'selector': 'td', 'props': [('padding', '8px')]}
                ])
                return styler

            styled_html = apply_custom_style(display_df.style).hide(axis='index').to_html()
            styled_html = styled_html.replace('<table', '<table style="width: 100%; border-collapse: collapse;"')
            st.markdown(styled_html, unsafe_allow_html=True)
            
            st.divider()
            
            st.subheader(f"📈 {district_name} 월별 매출 및 다음 달 수익 변동 추이")
            if not chart_data.empty:
                melted_df = chart_data.reset_index().melt(id_vars='결제 시기 (YYMM)', var_name='영업 시간대', value_name='월간 총매출 합산액')
                
                chart = alt.Chart(melted_df).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('결제 시기 (YYMM):N', 
                            title='연월 (YYMM)', 
                            axis=alt.Axis(labelAngle=0, labelFontSize=12, titlePadding=10, 
                                          labelColor='black', titleColor='black', titleFontSize=13,
                                          domainColor='black', tickColor='black')),
                    y=alt.Y('월간 총매출 합산액:Q', 
                            title='상권 월 매출액 (원)',
                            axis=alt.Axis(titlePadding=15, 
                                          labelColor='black', titleColor='black', titleFontSize=13,
                                          domainColor='black', tickColor='black',
                                          gridColor='#E2E2E2')),
                    color=alt.Color('영업 시간대:N', title='영업 시간대', 
                                    scale=alt.Scale(domain=["새벽", "아침", "오후", "밤"], 
                                                    range=["#9333ea", "#22c55e", "#f97316", "#2563eb"]),
                                    legend=alt.Legend(titleColor='black', labelColor='black')),
                    tooltip=['결제 시기 (YYMM)', '영업 시간대', '월간 총매출 합산액']
                ).properties(height=450)
                
                st.altair_chart(chart, use_container_width=True)