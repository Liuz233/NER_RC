import collections
from collections import Counter
import re


def read_predictions(file_list):
    """读取所有文件中的预测结果"""
    all_predictions = []
    for file in file_list:
        with open(file, 'r') as f:
            all_predictions.append([line.strip() for line in f])
    return all_predictions


def count_entities(predictions):
    """统计所有预测结果中的实体出现次数"""
    entity_counts = collections.defaultdict(int)
    for prediction in predictions:
        entities = prediction.split('/')
        for entity in entities:
            if entity:
                entity_counts[entity] += 1
    return entity_counts


def select_top_entities(entity_counts, top_n=2):
    """选择出现次数最多的top_n个实体，若出现次数相同则选择字符多者"""
    sorted_entities = sorted(entity_counts.items(), key=lambda x: (-x[1], -len(x[0])))
    return [entity for entity, count in sorted_entities[:top_n]]


def aggregate_predictions(file_lists, output_file):
    all_predictions = read_predictions(file_lists)
    """对所有预测结果进行集成，并输出到文件"""
    num_sentences = len(all_predictions[0])
    aggregated_results = []

    for i in range(num_sentences):
        sentence_predictions = [predictions[i] for predictions in all_predictions]
        entity_counts = count_entities(sentence_predictions)
        top_entities = select_top_entities(entity_counts)
        aggregated_results.append('/'.join(top_entities) + '/')

    with open(output_file, 'w') as f:
        for result in aggregated_results:
            f.write(result + '\n')


def test_acc(file_lists):
    entities = []
    with open("./data/SemEval2010_task8/original/test.txt", "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i + 1].strip()
            entity = re.findall(r'\((.*?)\)', line)[0].split(",")
            entities.append(entity)
    for file in file_lists:
        count = 0
        with open(file, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                pred_entities = lines[i].strip()[:-1].split("/")
                if Counter(entities[i]) == Counter(pred_entities):
                    count += 1
            print(f"{file} acc:{count / len(lines)}")


if __name__ == "__main__":
    data_path = "./output/"
    file_names = ["lstm.txt", "lstm.txt","bert-base-uncased.txt", "deberta-v3-large.txt", "new/bert-base-uncased.txt",
                  "new/deberta-v3-large.txt", "new/deberta-v3-large.txt","new/deberta-v3-large.txt"]  # 替换为你的文件列表
    file_lists = [data_path + item for item in file_names]
    outfile = f"./output/ensemble/5+2lstm.txt"

    aggregate_predictions(file_lists, outfile)

    file_lists.append("./output/ensemble/5+2lstm.txt")
    test_acc(file_lists)
