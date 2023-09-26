# Malicious traffic detection training based on deep learning.
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# training & validing
class IDS_MODEL():
    def __init__(self, model_name):
        super(IDS_MODEL, self).__init__()
        self.model = model_name.to(device)
        self.num_epochs = 6
        self.batch_size = 128
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self, data, target):
        data = torch.tensor(data).float().to(device)
        target = torch.tensor(target).long().to(device)
        train_data = TensorDataset(data, target)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.model.to(device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for text, label in train_dataloader:
                text, label = text, label
                self.optimizer.zero_grad()
                output = self.model(text)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}')
            with torch.no_grad():
                self.model.eval()
                all_predictions = []
                all_labels = []
                for text, label in train_dataloader:
                    text, label = text, label
                    output = self.model(text)
                    predictions = output.cpu().argmax(1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

                print('Validation ---epoch:{}/{}'.format(epoch, self.num_epochs))
                labels = list(set(all_labels))
                f1_scores = []
                precision_scores = []
                recall_scores = []
                for label in labels:
                    # Create a temporary binary label for the current category, setting that category label to 1 and other labels to 0
                    y_true_temp = [1 if y == label else 0 for y in all_labels]
                    y_pred_temp = [1 if y == label else 0 for y in all_predictions]

                    # f1,precision,recall
                    f1 = f1_score(y_true_temp, y_pred_temp)
                    precision = precision_score(y_true_temp, y_pred_temp)
                    recall = recall_score(y_true_temp, y_pred_temp)
                    # add result to list
                    f1_scores.append(f1)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                # output
                for i, label in enumerate(labels):
                    print(
                        f"classification {label}: F1 = {f1_scores[i]}, precision_scores = {precision_scores[i]}, recall_scores = {recall_scores[i]}")
        return self.model

    def predict(self, x_test):
        self.model.eval()
        pre_all = torch.tensor([])
        with torch.no_grad():
            test_dataloader = DataLoader(x_test, batch_size=128, shuffle=False)
            for batch, data in enumerate(test_dataloader):
                x_data = torch.tensor(data.float()).to(device)
                predict = self.model(x_data)
                pre = predict.cpu().detach().argmax(1)
                pre_all = torch.cat((pre_all, pre), dim=0)
        return pre_all.numpy()



# CNN-LSTM
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, kernel_size, dropout):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_dim, kernel_size=kernel_size)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = x.unsqueeze(1).expand(-1, 8, -1)
        embedded = embedded.permute(0, 2, 1)  # 调换维度，为卷积层做准备
        conved = F.relu(self.conv1(embedded))
        conved = conved.permute(0, 2, 1)  # 调换维度，为LSTM做准备
        output, (hidden, cell) = self.lstm(conved)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])  # 获取序列的最后一个时间步的输出作为分类器的输入
        return output


# RCNN
class RCNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, kernel_size, num_classes):
        super(TextRCNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_dim * 2 + input_dim, 256, kernel_size=3)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, 8, -1)
        lstm_out, _ = self.lstm(x)

        concat_features = torch.cat((x, lstm_out), dim=2)

        conv_out = F.relu(self.conv1(concat_features.transpose(1, 2)))
        conv_out = F.relu(self.conv2(conv_out))

        max_pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)

        logits = self.fc(max_pooled)
        return logits



