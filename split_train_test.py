import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/archive/sign_mnist/sign_mnist.csv')

# Разделяем данные на train, val, test с учетом пропорций для каждого класса
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, stratify=temp_df['label'], random_state=42)

# Печатаем размеры получившихся наборов данных
print("Train set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
