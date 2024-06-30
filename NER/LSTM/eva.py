# 读取模型预测结果文件和真实结果文件
with open('./data/result0.txt', 'r') as pred_file, open('./data/results.txt', 'r') as true_file:
    pred_lines = pred_file.readlines()
    true_lines = true_file.readlines()

# 去除行末尾的换行符
pred_lines = [line.strip() for line in pred_lines]
true_lines = [line.strip() for line in true_lines]

# 计算准确率
total = len(pred_lines)
correct = sum(1 for pred, true in zip(pred_lines, true_lines) if pred == true)
accuracy = correct / total

print(f"Accuracy: {accuracy:.2%}")