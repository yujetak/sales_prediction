import pandas as pd

df = pd.read_csv('./dataset/card_sales_summary.csv')

df_small = df[df['std_ym'] >= 202301].copy()
df_small.to_csv('./dataset/card_sales_summary_small.csv', index=False)

print("202301 이후 저장 완료")