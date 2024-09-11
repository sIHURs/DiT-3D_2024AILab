
import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns



# 定义正则表达式模式
epoch_pattern = re.compile(r'epoch_(\d+)\.pth')
data_pattern = re.compile(r'\{(.*)\}')

# 存储结果的列表
data_list_adapter = []
data_list_no_adapter = []
# 读取日志文件
file_path = 'checkpoints/validation_S4_chair_completion_finetune_adapter/output.log'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取 epoch 数值
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        
        # 提取大括号中的数据
        data_match = data_pattern.search(line)
        if data_match:
            data_str = data_match.group(1)
            try:
                data_dict = ast.literal_eval(f"{{{data_str}}}")
            except (SyntaxError, ValueError):
                data_dict = {}
            
            if epoch is not None:
                data_dict['epoch'] = epoch
                data_list_adapter.append(data_dict)
            # print(data_dict)
            # pd.concat([df, data_dict], ignore_index=True)

df_yes = pd.DataFrame(data_list_adapter)

file_path = 'checkpoints/validation_S4_chair_completion_finetune/output.log'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取 epoch 数值
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        
        # 提取大括号中的数据
        data_match = data_pattern.search(line)
        if data_match:
            data_str = data_match.group(1)
            try:
                data_dict = ast.literal_eval(f"{{{data_str}}}")
            except (SyntaxError, ValueError):
                data_dict = {}
            
            if epoch is not None:
                data_dict['epoch'] = epoch
                data_list_no_adapter.append(data_dict)
            # print(data_dict)
            # pd.concat([df, data_dict], ignore_index=True)

df_no = pd.DataFrame(data_list_no_adapter)

# 绘制线图
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_yes, x='epoch', y='1-NNA-CD', marker='o', color='b',label='fine tune with adaptformer')
sns.lineplot(data=df_no, x='epoch', y='1-NNA-CD', marker='o', color='g', label='global fine tune')
plt.xlabel('Epoch')
plt.ylabel('1-NNA-CD')
plt.title('1-NNA-CD over Epochs')
plt.legend()
plt.grid(True)
plt.show()