import pandas as pd

df = pd.read_csv(r'E:\CSI_CB\loan_qa_dataset_llama.csv')
print('First 5 rows:\n', df.head())
print('Total rows:', len(df))
print('Missing values:\n', df.isnull().sum())
print('Rows without <|END|>:\n', df[~df['answer'].str.contains(r'\<\|END\|\>', na=False)])
print('Unique questions:', len(df['question'].unique()))