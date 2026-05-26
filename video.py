import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 自定义数据集类
class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义一个简单的线性回归模型
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# 主函数
def main():
    # 读取 CSV 文件
    data = pd.read_csv('/Users/liushili/Desktop/vmd_gwo_lstm/波面统计.csv')
    # 假设第一列是日期列，删除它
    data = data.drop(data.columns[0], axis=1)
    # 假设最后一列为标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = CSVDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = CSVDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    input_size = X_train.shape[1]
    model = SimpleModel(input_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 记录损失值
    train_losses = []
    val_losses = []

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # 在验证集上评估模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
