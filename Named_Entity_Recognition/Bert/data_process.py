import re


# 处理Conll格式数据，将其转换为标签序列
def preprocess_conll2label(data_path):
    files = ["test.txt", "train.txt", "valid.txt"]
    for file in files:
        input_file = data_path + file
        output_file = data_path + f"processed_{file}"

        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            sentence = []
            bio_tags = []

            # 跳过第一行
            next(infile)

            for line in infile:
                if line.strip() == "":
                    if sentence:
                        outfile.write(" ".join(sentence) + "\n")
                        outfile.write(" ".join(bio_tags) + "\n")
                        sentence = []
                        bio_tags = []
                else:
                    parts = line.strip().split()
                    word = parts[0]
                    bio_tag = parts[3].split('-')[0]  # 只取BIO标签，忽略类别
                    sentence.append(word)
                    bio_tags.append(bio_tag)

            if sentence:
                outfile.write(" ".join(sentence) + "\n")
                outfile.write(" ".join(bio_tags) + "\n")


# 处理Sem数据集，生成二分类概率
def process_sem2prob_train(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()

        for i in range(0, len(lines), 2):
            sentence = lines[i].strip().replace("\"", "").replace(",", "")  # 去除句子中的引号和逗号
            if sentence.endswith('.'):
                sentence = sentence[:-1]  # 去除句子末尾的句号
            sentence = sentence.split(" ", 1)[1]  # 去除开头多余的字符
            if sentence[0] == " ":
                sentence = sentence[1:]

            relation = lines[i + 1].strip()
            relation_words = re.findall(r'\(([^,]+),([^)]+)\)', relation)  # 提取关系中的实体词对
            relation_words = [word.strip() for pair in relation_words for word in pair]  # 去除实体词对中的空格

            sentence_words = sentence.split()
            probabilities = [1 if word in relation_words else 0 for word in sentence_words]  # 生成概率序列

            outfile.write(f"{sentence}\n")
            outfile.write(f"{probabilities}\n")


def process_prob2prompt(input_file1: str, input_file2: str, output_file: str):
    with open(input_file1, 'r', encoding='utf-8') as infile1, open(input_file2, 'r', encoding='utf-8') as infile2, open(
            output_file, 'w', encoding='utf-8') as outfile:
        lines1 = infile1.readlines()
        lines2 = infile2.readlines()
        for i in range(0, len(lines1), 2):
            sentence = lines1[i].strip()
            probabilities = eval(lines1[i + 1].strip())
            label = lines2[int(i / 2)].strip()
            extended_sentence = f"Extract entities with relation {label} in sentence {sentence}"
            # 计算新增单词的数量
            original_words_count = len(sentence.split())
            extended_words_count = len(extended_sentence.split())
            new_words_count = extended_words_count - original_words_count

            # 扩展后的二分类标注
            extended_probabilities = [0] * new_words_count + probabilities
            outfile.write(f"{extended_sentence}\n")
            outfile.write(f"{extended_probabilities}\n")


def process_sem2prob_test(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        for line in lines:
            sentence = line.strip().split("	")[1].replace(".", "").replace("\"", "")
            entity1 = re.findall(r'<e1>(.*?)</e1>', sentence)[0]
            entity2 = re.findall(r'<e2>(.*?)</e2>', sentence)[0]
            sentence = sentence.replace("<e1>", "").replace("</e1>", "")
            sentence = sentence.replace("<e2>", "").replace("</e2>", "")
            sentence_words = sentence.split()
            probabilities = [1 if word in [entity1, entity2] else 0 for word in sentence_words]  # 生成概率序列
            outfile.write(f"{sentence}\n")
            outfile.write(f"{probabilities}\n")


if __name__ == "__main__":
    # 调用函数处理文件
    # process_sem2prob_train('./data/SemEval2010_task8/given/train.txt',
    #                        './data/SemEval2010_task8/prob/train.txt')  # 处理关系抽取数据并生成概率
    # process_sem2prob_test('./data/SemEval2010_task8/original/test_1.txt',
    #                       './data/SemEval2010_task8/prob/test.txt')  # 处理关系抽取数据并生成概率
    process_prob2prompt("./data/SemEval2010_task8/prob/train.txt", "./data/SemEval2010_task8/given/train_seq.txt",
                        "./data/SemEval2010_task8/prompt/train.txt")
    process_prob2prompt("./data/SemEval2010_task8/prob/test.txt", "./data/SemEval2010_task8/given/test_seq_deberta-v3-large.txt",
                        "./data/SemEval2010_task8/prompt/test_deberta-v3-large.txt")
