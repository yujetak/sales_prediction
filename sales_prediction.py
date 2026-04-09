import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

# https://data.gg.go.kr/portal/data/service/selectServicePage.do?page=1&rows=10&sortColumn=&sortDirection=&infId=88UZJGHAP6WFG5TSOBP338234799&infSeq=2&order=&loc=&srvCd=F&cateId=GG29
path = './dataset/card_sales_summary.csv'

df = pd.read_csv(path)
df.head(3)

# 하루를 시간대별로 나누기 (영업시간과 더 연관성이 생기도록)
def compute_tmzone(df):
  tm = int(df['tmzon_cd'][-2:])
  if (tm > 8):
    return 'night'
  elif (tm > 5):
    return 'afternoon'
  elif (tm > 2):
    return 'morning'
  else:
    return 'dawn'

# 데이터셋에서 하루 매출 총합 row 지우기
df = df[df['tmzon_cd'] != 'TOT'].copy()

# 기간년월, 시군구코드, 시간대코드(TZ01~TZ10, 하루를 10개의 시간대로 나눔)
# 중분류업종코드, 매출금액(label)
selected_cols = ['std_ym', 'signgu_cd', 'tmzon_cd', \
    'mdclass_indutype_cd', 'sales_amt']

# 기간년월 연도와 월로 분리
df['year'] = df['std_ym'] // 100
df['month'] = df['std_ym'] % 100

# 생활시간대로 재구성(아침, 점심, 저녁, 새벽)
df['tmzon_group'] = df.apply(compute_tmzone, axis = 1)

sigungu_categories = sorted(df['signgu_cd'].unique().tolist())
tz_categories = sorted(df['tmzon_cd'].unique().tolist())
mdclass_catetories = sorted(df['mdclass_indutype_cd'].unique().tolist())
tz_group_catetgories = ['dawn', 'morning', 'afternoon', 'night']

oe = OrdinalEncoder(categories=[sigungu_categories, tz_categories, \
                                mdclass_catetories, tz_group_catetgories])
oe_cols = ['signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd','tmzon_group']
df[oe_cols] = oe.fit_transform(df[oe_cols])

# skewed data 로그변환
df['sales_amt_log'] = np.log1p(df['sales_amt'])
# 12월과 1월이 가깝다는 것(주기)을 표현하기 위한 cyclic encoding
df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)

# 사용할 feature의 값들을 맞추기 위해 스케일링
ss_cols = ['year', 'signgu_cd', 'tmzon_cd', 'mdclass_indutype_cd', \
               'tmzon_group']
ss = StandardScaler()
df[ss_cols] = ss.fit_transform(df[ss_cols])

cols = df.columns.tolist()

# 최종 사용할 feature 목록
feature_cols = ['signgu_cd', 'mdclass_indutype_cd',
                'tmzon_group', 'year', 'month_sin', 'month_cos', 'sales_amt_log']


#필요없는 feature 모두 제거
df.drop([
    'tmzon_cd', 'sales_amt', 'sales_amt_rate', 'bfym_incndecr_val',\
    'bfym_incndecr_rate', 'bfyy_smmn_incndecr_val', 'bfyy_smmn_incndecr_rate']\
        , axis=1, inplace=True)

#상관관계 분석
import seaborn as sns
import matplotlib.pyplot as plt

corr = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
plt.show()

# 연단위로 시계열 분석하기
time_step = 12

X_train, y_train, X_test, y_test = [], [], [], []

for _, group in df.groupby(['signgu_cd', 'mdclass_indutype_cd', 'tmzon_group']):
    group = group.sort_values('std_ym')
    if len(group) <= time_step:
        continue
    values = group[feature_cols].values
    target = group['sales_amt_log'].values

    sequences_X, sequences_y = [], []
    for i in range(len(values) - time_step):
        sequences_X.append(values[i:i+time_step])
        sequences_y.append(target[i+time_step])

    split = int(len(sequences_X) * 0.8)
    X_train.extend(sequences_X[:split])
    y_train.extend(sequences_y[:split])
    X_test.extend(sequences_X[split:])
    y_test.extend(sequences_y[split:])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
input_shape = (X_train.shape[1], X_train.shape[2])

# 학습 및 예측
def train_and_predict(model_name, batch_size=512, optimizer='adamw'):
    model = Sequential()

    # batch_size 32, 64, 128
    # if model_name == 'RNN':
    #     model.add(SimpleRNN(64, return_sequences=False, input_shape=input_shape))
    if model_name == 'LSTM':
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    elif model_name == 'BiLSTM':
        model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)))
        model.add(Bidirectional(LSTM(64, return_sequences=False, input_shape=input_shape)))
    elif model_name == 'GRU':
        model.add(GRU(64, return_sequences=False, input_shape=input_shape))

    # LayerNormalization, BatchNormalization, GroupNormalization
    model.add(LayerNormalization())
    model.add(Dropout(0.4))

    # Hidden Layer 1
    model.add(Dense(64, activation='tanh'))
    model.add(LayerNormalization())
    model.add(Dropout(0.4))

    # Hidden Layer 2
    model.add(Dense(32, activation='tanh'))

    # Output Layer
    model.add(Dense(1))

    opt = AdamW(learning_rate=0.0001, weight_decay=1e-4)
    model.compile(optimizer=opt, loss='mse')

    os.makedirs('./models', exist_ok=True)
    checkpoint_path = './models/model.keras'
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

    print(f"\n[Training] {model_name} | batch={batch_size} | optimizer={optimizer}")
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=10,
                        batch_size=batch_size,
                        callbacks=[checkpoint, early_stop],
                        verbose=1)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    pred_original = np.expm1(pred)

    return model, mae, rmse, r2, pred_original, history

# model_names = ['GRU', BiLSTM', 'LSTM']
# 최종모델 BiLSTM
model_names = ['BiLSTM']
models = {}
results = {}
predictions = {}
histories = {}

for m in model_names:
    key = f"{m}"
    model, mae, rmse, r2, pred, history = train_and_predict(m)
    models[key] = model
    results[key] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    predictions[key] = pred
    histories[key] = history
    print(f"[{key}] MAE: {mae:.4f} RMSE: {rmse:.4f} | R2: {r2:.4f}")

print("\n=== 결과 요약 ===")
for key, metric in sorted(results.items(), key=lambda x: x[1]['rmse']):
    print(f"{key}: MAE={metric['mae']:.4f} | RMSE={metric['rmse']:.4f} | R2={metric['r2']:.4f}")

# 인코더, 스케일러 파일로 저장
import joblib
import os

os.makedirs('./models', exist_ok=True)
joblib.dump(oe, './models/oe.pkl')
joblib.dump(ss, './models/ss.pkl')