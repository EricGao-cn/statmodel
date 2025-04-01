import pandas as pd

df = pd.read_excel('data/short_comments_cleaned.xlsx')
positive_df = df[df['情感分类'] == '积极']
positive_count = len(positive_df)
positive_avg_score = positive_df['sentiment_score'].mean()

print(f"积极情感的数据条数: {positive_count}")
print(f"积极情感的平均sentiment_score: {positive_avg_score:.4f}")

negative_df = df[df['情感分类'] == '消极']
negative_count = len(negative_df)
negative_avg_score = negative_df['sentiment_score'].mean()

print(f"消极情感的数据条数: {negative_count}")
print(f"消极情感的平均sentiment_score: {negative_avg_score:.4f}")