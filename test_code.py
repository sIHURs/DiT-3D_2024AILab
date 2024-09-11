import re
import pandas as pd
import matplotlib.pyplot as plt

# 日志数据
log_data = """
2024-08-22 11:03:01,314 :128 {'CD': tensor(0.0051, device='cuda:0'), 'EMD': tensor(73.4363, device='cuda:0'), 'fscore1': tensor(0.5174)}
2024-08-22 11:04:56,759 :256 {'CD': tensor(0.0033, device='cuda:0'), 'EMD': tensor(61.7158, device='cuda:0'), 'fscore1': tensor(0.6379)}
2024-08-22 11:06:23,745 :512 {'CD': tensor(0.0028, device='cuda:0'), 'EMD': tensor(57.3172, device='cuda:0'), 'fscore1': tensor(0.6759)}
2024-08-22 11:08:38,852 :1024 {'CD': tensor(6.2083, device='cuda:0'), 'EMD': tensor(13042.4736, device='cuda:0'), 'fscore1': tensor(0.6775)}
"""

# 正则表达式来提取数据
pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) :\d+ {'CD': tensor\((?P<cd>[0-9.]+), device='cuda:0'\), 'EMD': tensor\((?P<emd>[0-9.]+), device='cuda:0'\), 'fscore1': tensor\((?P<fscore>[0-9.]+)\)}"
)

# 用来存储数据的列表
data = []

# 对每一行进行匹配
for match in pattern.finditer(log_data):
    timestamp = match.group("timestamp")
    cd = float(match.group("cd"))
    emd = float(match.group("emd"))
    fscore = float(match.group("fscore"))
    data.append({"timestamp": timestamp, "CD": cd, "EMD": emd, "fscore1": fscore})

# 打印结果
df = pd.DataFrame(data)

print(df)

# 生成一个从 256 到 1792，间隔为 256 的 Series
index_series = pd.Series([128, 256, 512, 1024], name="Index")

# 假设 df 是之前的 DataFrame
# 确保 df 中的行数与 index_series 一致
# if len(df) != len(index_series):
#     raise ValueError("The length of the index_series does not match the length of the DataFrame.")

# 将 index_series 与 df 中的 CD 列合并到一个新的 DataFrame
new_df = pd.DataFrame({
    "Index": index_series,
    "CD": df["CD"]
})

# 打印新的 DataFrame
print(new_df)

plt.figure(figsize=(10, 6))
plt.plot(new_df["Index"], new_df["CD"], marker='o', linestyle='-', color='b')
plt.title('different levels sampling from input sparse point cloud - case2')
plt.xlabel('number of points from input sparse point cloud')
plt.ylabel('chamfer distance')

plt.xticks(ticks=range(256, 1793, 256))
plt.grid(True)
plt.show()


