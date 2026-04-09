import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Setup page
st.set_page_config(page_title="카드 매출 예측 및 변동성 시각화", layout="wide", page_icon="💳")
st.title("💸 카드 매출 시뮬레이션 및 Top3 동향 집중 분석")
st.markdown("유동인구 및 소비자의 시간대별 소비 성향을 파악하여 매장 운영시간 조정, 마케팅 타이밍 전략 수립 등 실무 활용에 적합합니다.")
st.markdown("---")

TIMEZONE_MAP = {
    'dawn': '새벽(0시~6시)',
    'morning': '오전(6시~12시)',
    'afternoon': '오후(12시~19시)',
    'night': '저녁(19시~24시)',
    'TOT': '전체 시간대 평균 (일단위 통합)'
}
INV_TIMEZONE_MAP = {v: k for k, v in TIMEZONE_MAP.items()}

def format_korean_currency(v):
    if pd.isna(v) or v == 0: return "0원"
    is_neg = v < 0
    v = abs(int(v))
    
    eok = v // 100000000
    rem = v % 100000000
    man = rem // 10000
    cheon = rem % 10000
    
    parts = []
    if eok > 0: parts.append(f"{eok:,}억")
    if man > 0: parts.append(f"{man:,}만")
    if cheon > 0 and eok == 0 and man == 0: parts.append(f"{cheon:,}")
    if len(parts) == 0: parts.append("0")
    
    res = " ".join(parts) + "원"
    return f"-{res}" if is_neg else res

def format_percentage(v):
    if pd.isna(v): return "0%"
    return f"{v:+.1f}%"

@st.cache_resource
def load_resources():
    import os
    model_path = './models/model.keras'
    oe_path = './models/oe.pkl'
    ss_path = './models/ss.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(oe_path) or not os.path.exists(ss_path):
        return None, None, None, None, None

    with open('./sources/mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        
    with open('./sources/region_mapping.json', 'r', encoding='utf-8') as f:
        region_mapping = json.load(f)
        
    oe = joblib.load(oe_path)
    ss = joblib.load(ss_path)
        
    import tensorflow as tf
    # 호환성과 관련된 경고 이슈 방지를 위해 compile=False 사용할 수도 있습니다.
    model = tf.keras.models.load_model(model_path, compile=False)
        
    return mapping, region_mapping, oe, ss, model

def format_ym(ym_val):
    ym_str = str(int(ym_val))
    if len(ym_str) == 6:
        return f"{ym_str[:4]}년 {ym_str[4:]}월"
    return str(ym_val)

with st.spinner("데이터 로딩 중..."):
    mapping, region_mapping, oe, ss, model = load_resources()
    
    if model is None:
        st.error("🚀 모델 또는 스케일러 파일이 없습니다. sales_prediction.py로 모델을 훈련하고 생성해주세요.")
        st.stop()
        
    df = load_data()

# Data basic preparations
df['year'] = df['std_ym'] // 100
df['month'] = df['std_ym'] % 100
df['포맷된연월'] = df['std_ym'].apply(format_ym)

def compute_tmzone(tmzon_cd):
    if tmzon_cd == 'TOT':
        return 'TOT'
    tm_str = str(tmzon_cd)
    if 'TOT' in tm_str: return 'TOT'
    try:
        tm = int(tm_str[-2:])
        if (tm > 8): return 'night'
        elif (tm > 5): return 'afternoon'
        elif (tm > 2): return 'morning'
        else: return 'dawn'
    except:
        return 'TOT'

df['tmzon_group_raw'] = df['tmzon_cd'].apply(compute_tmzone)

# Get original unique values from encoders
sigungu_options = sorted(df['signgu_cd'].unique().tolist())
mdclass_options = sorted(df['mdclass_indutype_cd'].unique().tolist())
tmzon_group_korean_options = list(TIMEZONE_MAP.values())

def map_industry(code): return mapping.get(str(code), str(code))
def map_region(code): return region_mapping.get(str(code), str(code))

tab1, tab2 = st.tabs(["🚀 맞춤형 매출 예측 및 트렌드", "🏆 연도별 변동률 Top 3 추적 대시보드"])

with tab1:
    st.header("특정 지역/시간대/업종의 맞춤형 향후 매출 예측")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_names = [f"{map_region(c)} ({c})" for c in sigungu_options]
        selected_region_name = st.selectbox("📌 행정구역 선택", options=region_names)
        selected_region_code = int(selected_region_name.split('(')[-1].replace(')', ''))
        
    with col2:
        selected_timezone_kor = st.selectbox("🕰️ 시간대 확인 범위", options=tmzon_group_korean_options)
        selected_timezone_group = INV_TIMEZONE_MAP[selected_timezone_kor]
        
    with col3:
        industry_names = [f"{map_industry(c)} ({c})" for c in mdclass_options]
        selected_industry_name = st.selectbox("🏬 관심 업종 선택", options=industry_names)
        idx = industry_names.index(selected_industry_name)
        selected_industry_code = mdclass_options[idx]
        
    if st.button("탐색 및 예측 생성", use_container_width=True, type="primary"):
        origin_val_name = selected_industry_name.split('(')[0].strip()
        
        target_df = df[(df['signgu_cd'] == selected_region_code) & 
                       (df['mdclass_indutype_cd'] == selected_industry_code)].copy()
                       
        if selected_timezone_group != 'TOT':
            target_df = target_df[target_df['tmzon_group_raw'] == selected_timezone_group]
        
        agg_df = target_df.groupby('std_ym').agg({
            'sales_amt': 'mean',
            'year': 'first',
            'month': 'first',
            '포맷된연월': 'first',
            'tmzon_group_raw': 'first'
        }).reset_index()
        
        agg_df = agg_df.sort_values('std_ym')
        
        if len(agg_df) < 12 and selected_timezone_group != 'TOT':
            st.warning(f"선택한 조건의 과거 데이터가 12건 미만이라 예측 수행이 불가합니다.")
        
        # 모델 예측 로직
        if selected_timezone_group != 'TOT' and len(agg_df) >= 12:
            input_df = agg_df.tail(12).copy()
            input_df['tmzon_cd'] = 'TZ01' # 더미
            input_df['signgu_cd'] = selected_region_code
            input_df['mdclass_indutype_cd'] = selected_industry_code
            input_df['tmzon_group'] = input_df['tmzon_group_raw']
            
            oe_cols = ['signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd','tmzon_group']
            input_df[oe_cols] = oe.transform(input_df[oe_cols])
            
            input_df['sales_amt_log'] = np.log1p(input_df['sales_amt'])
            input_df['month_sin'] = np.sin(2 * np.pi * input_df['month']/12.0)
            input_df['month_cos'] = np.cos(2 * np.pi * input_df['month']/12.0)
            
            ss_cols = ['year', 'signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', 'tmzon_group']
            input_df[ss_cols] = ss.transform(input_df[ss_cols])
            
            feature_cols = ['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group', 'year', 'month_sin', 'month_cos', 'sales_amt_log']
            
            seq_x = input_df[feature_cols].values
            seq_x = np.expand_dims(seq_x, axis=0)
            
            pred_log = model.predict(seq_x, verbose=0)
            pred_val = np.expm1(pred_log)[0][0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(label=f"💰 [{origin_val_name}] 다음 달 예상 매출 금액", value=format_korean_currency(pred_val))
            st.markdown("<br>", unsafe_allow_html=True)
            
        elif selected_timezone_group == 'TOT':
            st.info("💡 하루 전체 평균(TOT)은 과거 트렌드 시각화만 제공되며 모델의 예측 대상이 아닙니다.")
            
        st.subheader(f"📊 [{origin_val_name}] 매출 추세 현황")
        
        if not agg_df.empty:
            agg_df['매출액(텍스트)'] = agg_df['sales_amt'].apply(format_korean_currency)
            agg_df['연도'] = agg_df['year'].astype(str)
            fig = px.line(agg_df, 
                          x='month', 
                          y='sales_amt', 
                          color='연도',
                          markers=True, 
                          labels={'sales_amt': '평균 매출금액(원)', 'month': '조사 월 단위', '연도': '연도'},
                          title=f"[{origin_val_name}] 연도별 매출 추세 비교",
                          hover_data={"포맷된연월": True, "sales_amt": False, "매출액(텍스트)": True})
            
            fig.update_layout(xaxis=dict(tickmode='linear', dtick=1, range=[1,12]), xaxis_title="월", yaxis_title="월 평균 매출")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🏆 콤보 차트: 연도별 변동률 Top 3 사업 정밀 추적")
    st.markdown("특정 연도에 변화율(증감/감소)이 가장 두드러진 최고/최저 사업 3가지를 도출하고, 각각의 금액(Bar)과 변동률(Line)을 한 차트에 비교합니다.")
    
    # 탭 2 집계 시에는 반드시 'TO'(업종전체), 'TOT'(전체시간대) 등 합계용 코드를 원천 제거합니다!
    tab2_df = df[(df['mdclass_indutype_cd'] != 'TO') & (df['tmzon_cd'] != 'TOT')].copy()
    
    # 시간대별 최고 매출 업종 가이드 섹션
    st.markdown("#### 🕒 시간대별 최고 평균 매출을 올리는 인기 업종 (업종전체 제외)")
    tz_guide_cols = st.columns(4)
    for idx, tz in enumerate(['dawn', 'morning', 'afternoon', 'night']):
        with tz_guide_cols[idx]:
            tz_df = tab2_df[tab2_df['tmzon_group_raw'] == tz]
            if not tz_df.empty:
                top_ind = tz_df.groupby('mdclass_indutype_cd')['sales_amt'].mean().idxmax()
                st.info(f"**{TIMEZONE_MAP[tz]}**\n\n🏆 1위: **{map_industry(top_ind)}**")
    st.divider()

    st.markdown("### 📅 연도별 통합 랭킹 대시보드 (2018 ~ 최신)")
    st.markdown("복잡한 월별 차트 대신, **매년 가장 압도적인 매출을 올린 상권(지역+업종)**과 **전년 대비 가장 폭발적으로 성장한 상권** Top 3를 직관적으로 비교합니다.")
    
    years = sorted(tab2_df['year'].dropna().unique().tolist(), reverse=True)
    
    sales_top_data, sales_worst_data = [], []
    growth_top_data, growth_worst_data = [], []
    
    for y in years:
        y_df = tab2_df[tab2_df['year'] == y]
        
        # 1. 💰 매출액 분석 (최고/최저)
        sales_g = y_df.groupby(['signgu_cd', 'mdclass_indutype_cd'])['sales_amt'].mean().reset_index()
        if not sales_g.empty:
            sales_top3 = sales_g.nlargest(3, 'sales_amt')
            sales_worst3 = sales_g.nsmallest(3, 'sales_amt')
            
            s_top_row, s_worst_row = {"연도": f"{int(y)}년"}, {"연도": f"{int(y)}년"}
            
            for rank, (_, row) in enumerate(sales_top3.iterrows(), 1):
                label = f"{map_region(row['signgu_cd'])} - {map_industry(row['mdclass_indutype_cd'])}"
                s_top_row[f"🏆 {rank}위"] = f"{label}\n({format_korean_currency(row['sales_amt'])})"
            sales_top_data.append(s_top_row)
            
            for rank, (_, row) in enumerate(sales_worst3.iterrows(), 1):
                label = f"{map_region(row['signgu_cd'])} - {map_industry(row['mdclass_indutype_cd'])}"
                s_worst_row[f"🔻 최하 {rank}위"] = f"{label}\n({format_korean_currency(row['sales_amt'])})"
            sales_worst_data.append(s_worst_row)
            
        # 2. 🚀 성장률 분석 (최고/최저, NaN 제거)
        growth_df = y_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['bfym_incndecr_rate'])
        if not growth_df.empty:
            growth_g = growth_df.groupby(['signgu_cd', 'mdclass_indutype_cd'])['bfym_incndecr_rate'].mean().reset_index()
            growth_top3 = growth_g.nlargest(3, 'bfym_incndecr_rate')
            growth_worst3 = growth_g.nsmallest(3, 'bfym_incndecr_rate')
            
            g_top_row, g_worst_row = {"연도": f"{int(y)}년"}, {"연도": f"{int(y)}년"}
            
            for rank, (_, row) in enumerate(growth_top3.iterrows(), 1):
                label = f"{map_region(row['signgu_cd'])} - {map_industry(row['mdclass_indutype_cd'])}"
                g_top_row[f"🔥 {rank}위"] = f"{label}\n({format_percentage(row['bfym_incndecr_rate'])})"
            growth_top_data.append(g_top_row)
            
            for rank, (_, row) in enumerate(growth_worst3.iterrows(), 1):
                label = f"{map_region(row['signgu_cd'])} - {map_industry(row['mdclass_indutype_cd'])}"
                g_worst_row[f"📉 최하 {rank}위"] = f"{label}\n({format_percentage(row['bfym_incndecr_rate'])})"
            growth_worst_data.append(g_worst_row)
            
    st.markdown("---")
    st.markdown("##### 💰 연도별 [최고 매출액] 랭킹 Top 3")
    if sales_top_data:
        st.dataframe(pd.DataFrame(sales_top_data), use_container_width=True, hide_index=True)
        
    st.markdown("##### 💸 연도별 [최저 매출액] 랭킹 최하 3")
    if sales_worst_data:
        st.dataframe(pd.DataFrame(sales_worst_data), use_container_width=True, hide_index=True)
        
    st.markdown("##### 🚀 연도별 [최고 성장률] 랭킹 Top 3")
    if growth_top_data:
        st.dataframe(pd.DataFrame(growth_top_data), use_container_width=True, hide_index=True)
        
    st.markdown("##### 🚨 연도별 [최저 성장률(급락)] 랭킹 최하 3")
    if growth_worst_data:
        st.dataframe(pd.DataFrame(growth_worst_data), use_container_width=True, hide_index=True)

