# 读取原始文件并提取a, b，并存储到另一个文件中
with open('./data/sentences.txt', 'r') as input_file, open('./data/result0.txt', 'w') as output_file:
    for line in input_file:
        # 使用制表符分割每行的记录
        parts = line.split('\t')
        if len(parts) >= 3:
            a = parts[0].strip()
            b = parts[1].strip()
            # 将a和b用逗号连接并写入到结果文件中
            output_file.write(f'{a},{b}\n')