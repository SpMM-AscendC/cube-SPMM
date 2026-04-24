import pandas as pd

# 读取数据，用 | 分隔
df = pd.read_csv('coldense_Pajek.txt', sep='|', header=None)

# 使用您提供的列名
df.columns = ['Category', 'Sample_Name', 'M', 'K', 'N', 'NNZ', 
              'WindowNum', 'BlockNum', 'FillRate', 'Performance(us)']

# 去除可能的空格
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 保存为CSV
df.to_csv('coldense_Pajek.csv', index=False)
print("数据已保存为 data.csv")