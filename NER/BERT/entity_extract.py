import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 初始化
# model_name = "deberta-v3-large"  # batch_size = 64
model_name = "bert-base-uncased"  # batch_size = 128

batch_size = 128
if model_name == "deberta-v3-large":
    batch_size = 64
lr = 5e-5
epochs = 10
max_len = 48
num_labels = 2
if_prompt = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/root/model/" + model_name
model_save_dir = "./model/" + model_name
if if_prompt:
    model_save_dir = "./new_model/" + model_name

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels).to(device)


def set_seed(seed=42, use_cuda=True):
    # 设定PyTorch的随机种子
    torch.manual_seed(seed)
    # 如果使用CUDA
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU环境
    # 设定Python内置的随机种子
    random.seed(seed)
    # 设定Numpy的随机种子
    np.random.seed(seed)


# 定义数据集类
class EntityDataset(Dataset):
    def __init__(self, filename):
        global tokenizer, device
        self.sentences = []
        self.word_labels = []  # 保存单词级别的标签
        self.tokenizer = tokenizer
        self.device = device
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                text = lines[i].strip()
                label = list(map(int, lines[i + 1].strip().strip('[]').split(', ')))
                self.sentences.append(text)
                self.word_labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.word_labels[idx]

        # 使用tokenizer编码句子
        encoding = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True,
                                  max_length=max_len)
        input_ids = encoding['input_ids'].squeeze().to(self.device)
        attention_mask = encoding['attention_mask'].squeeze().to(self.device)

        # 创建一个与input_ids长度相同的标签数组
        labels = [-100] * input_ids.size(0)  # 使用-100作为忽略的标签值
        word_ids = encoding.word_ids()  # 获取每个token对应的单词索引

        previous_word_idx = None
        label_index = 0
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None:
                if word_index != previous_word_idx:  # 开始新单词的tokens
                    label_index = word_index
                # 将单词的标签分配给每个token
                if label_index < len(word_labels):
                    labels[token_index] = word_labels[label_index]
                previous_word_idx = word_index

        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        return input_ids, attention_mask, labels


def train_epoch(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    for input_ids, attention_mask, labels in tqdm(data_loader, desc=f"Epoch{epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=2)
        predictions.extend(preds.cpu().flatten().tolist())
        true_labels.extend(labels.cpu().flatten().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, accuracy


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=2)
            predictions.extend(preds.cpu().flatten().tolist())
            true_labels.extend(labels.cpu().flatten().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, accuracy


def test_case(model, text):
    sentence = text.split()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = (outputs.logits)[0][:len(sentence)]
        preds = torch.argmax(logits, dim=1)
        _, idx = torch.topk(preds, 2)
        entities = []
        for item in idx:
            entities.append(sentence[item])
    return entities


def test_file(in_file, model_path, out_file):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    count = 0
    tokens = None
    with open(in_file, "r", encoding="utf-8") as f:
        with open(out_file, "w", encoding="utf-8") as f_out:
            lines = f.readlines()
            for i in tqdm(range(0, len(lines), 2)):
                temp = ""
                line = lines[i].strip()
                sentence = line.split()
                label = eval(lines[i + 1].strip())
                entities = [sentence[i] for i in range(len(label)) if label[i] == 1]
                entities = [item.lower() for item in entities]
                pred_entities, tokens, predictions = predict_entities(line, model)
                if model_name == "deberta-v3-large":
                    for j in range(len(pred_entities)):
                        pred_entities[j] = pred_entities[j][1:]
                for j in range(len(pred_entities)):
                    if j <= 1:
                        temp += pred_entities[j] + "/"
                f_out.write(temp + "\n")
                # print(f"{entities}")
                # print(f"{pred_entities}")
                # print("---------------")
                if Counter(entities) == Counter(pred_entities):
                    count += 1
                else:
                    # if len(entities) == 2 and len(pred_entities) == 2:
                    print(f"{entities}")
                    print(f"{pred_entities}")
                    print(tokens)
                    print(predictions)
                    print("---------------")

    print(count / (len(lines) / 2))


def predict_entities(sentence, model):
    global tokenizer
    # 对句子进行编码
    encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 解析预测结果
    predictions = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 从标记和预测中提取实体
    entities = []
    current_entity = []
    for token, prediction in zip(tokens, predictions[0]):
        if token == '[PAD]':
            continue
        if token == '[SEP]':
            continue
        if prediction == 1:  # 假设 1 表示实体
            if token.startswith("##"):
                current_entity.append(token[2:])  # 移除 BPE 的 '##' 并添加到当前实体
            else:
                if current_entity:  # 如果当前实体列表非空，则先将之前的实体组合后加入到entities中
                    entities.append("".join(current_entity))
                    current_entity = []  # 开始新的实体
                current_entity.append(token)
        else:
            if current_entity:  # 在非实体标签处结束当前实体的组合
                entities.append("".join(current_entity))
                current_entity = []

    if current_entity:  # 确保添加最后一个实体
        entities.append("".join(current_entity))
    return entities, tokens, predictions


def main(if_save_per_epoch, if_prompt):
    data_path = "./data/SemEval2010_task8/prob/"
    if if_prompt:
        data_path = "./data/SemEval2010_task8/prompt/"
    train_dataset = EntityDataset(data_path + 'train.txt')
    if if_prompt:
        test_dataset = EntityDataset(data_path + f'test_{model_name}.txt')
    else:
        test_dataset = EntityDataset(data_path + f'test.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # print(test_dataset[0])
    # print(tokenizer.convert_ids_to_tokens(test_dataset[0][0]))

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_acc = 0
    best_epoch = 0
    best_dict = None
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if if_save_per_epoch:
            model_save_path = os.path.join(model_save_dir, f'epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_save_path)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_dict = model.state_dict()
        print(
            f"Epoch {epoch + 1}\n Train Loss: {train_loss}, Train Acc: {train_acc}\n Test Loss: {test_loss}, Test Acc: {test_acc}\n ")

    print(f"Best Epoch:{best_epoch + 1}")
    model_save_path = os.path.join(model_save_dir, f'best_model.pt')
    torch.save(best_dict, model_save_path)
    # 绘制训练集和测试集的loss曲线
    fig, ax1 = plt.subplots()

    # 绘制损失曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(train_losses, label='Train Loss', color='red')
    ax1.plot(test_losses, label='Test Loss', color='darkred')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # 将图例放置在图外的左侧
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

    # 创建共享x轴的第二个坐标轴，用于准确率
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(test_accs, label='Test Accuracy', color='darkblue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # 将图例放置在图外的右侧
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 0.85))

    plt.title('Training and Test Loss and Accuracy Over Time')
    fig.tight_layout()  # 调整整体布局以避免标签被剪裁
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    # main(False, if_prompt)
    test_file("./data/SemEval2010_task8/.txt", f"./model/{model_name}/" + "best_model.pt",
              f"./output/{model_name}.txt")
