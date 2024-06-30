import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 初始化
batch_size = 64
lr = 5e-5
epochs = 10
max_len = 48
num_labels = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deberta-v3-large"  # batch_size = 64
# model_name = "bert-base-uncased"  # batch_size = 128
model_path = "/root/model/" + model_name
model_save_dir = "./new_model/" + model_name

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)

label2idx = {
    "Other": 0,
    "Cause-Effect": 1,
    "Component-Whole": 2,
    "Entity-Destination": 3,
    "Product-Producer": 4,
    "Entity-Origin": 5,
    "Member-Collection": 6,
    "Message-Topic": 7,
    "Content-Container": 8,
    "Instrument-Agency": 9
}

# 反转字典的键和值
idx2label = {v: k for k, v in label2idx.items()}


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
        self.sentence_labels = []
        self.tokenizer = tokenizer
        self.device = device
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                sentence = lines[i].strip().replace("\"", "").replace(",", "")
                if sentence.endswith('.'):
                    sentence = sentence[:-1]
                sentence = sentence.split(" ", 1)[1]
                if sentence[0] == " ":
                    sentence = sentence[1:]

                label = lines[i + 1].strip().split("(")[0]
                self.sentences.append(sentence)
                self.sentence_labels.append(label2idx[label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.sentence_labels[idx]
        encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        input_ids = encoding['input_ids'].squeeze().to(self.device)
        attention_mask = encoding['attention_mask'].squeeze().to(self.device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, label_tensor


def train_epoch(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    for input_ids, attention_mask, labels in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())
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
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, accuracy


def test_file(in_file, model_path, out_file):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    count = 0
    with open(in_file, "r", encoding="utf-8") as f:
        with open(out_file, "w", encoding="utf-8") as f_out:
            lines = f.readlines()
            for i in tqdm(range(0, len(lines), 2)):
                line = lines[i].strip().replace(".", "")[:-1]
                label = lines[i + 1].strip().split("(")[0]
                # 对句子进行编码
                encoding = tokenizer(line, return_tensors='pt', padding='max_length', truncation=True,
                                     max_length=max_len)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                # 预测
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                pred_label = idx2label[pred.cpu().numpy()[0]]
                # print(label)
                # print(pred_label)
                if label == pred_label:
                    count += 1
                f_out.write(pred_label + "\n")
            print(count / (len(lines)/2))


def main(if_save_per_epoch):
    data_path = "./data/SemEval2010_task8/original/"
    train_dataset = EntityDataset(data_path + 'train.txt')
    test_dataset = EntityDataset(data_path + 'test.txt')
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
    model_save_path = os.path.join(model_save_dir, f'seq_best_model.pt')
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
    # main(False)
    test_file("./data/SemEval2010_task8/original/test.txt", f"./new_model/{model_name}/" + "seq_best_model.pt",
              f"./data/SemEval2010_task8/given/test_seq_{model_name}.txt")
