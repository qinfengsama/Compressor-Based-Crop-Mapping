import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Normalize
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import random
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from paddle import nn
import paddle.nn.functional as F

def main(args):
    # 使用 args 中的参数来替换原来直接写在代码中的部分
    select_gt_path = args.select_gt_path
    select_rs_path = args.select_rs_path
    train_ratio = args.train_ratio
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_path = args.save_path

def getTrainTest(label, trainNum):
    # 设置随机种子
    random.seed(32)
    testGt = copy.deepcopy(label)
    trainGt = np.zeros(label.shape)
    labelNum = int(max(label.ravel())) + 1
    for lab in range(1, labelNum):
        labIndex = np.where(label == lab)
        randomList = random.sample(range(0, len(labIndex[0])), int(len(labIndex[0])*trainNum))
        for randInt in randomList:
            x = labIndex[0][randInt]
            y = labIndex[1][randInt]
            trainGt[x][y] = lab
            testGt[x][y] = 0
    trainGt = trainGt.astype(int)
    testGt = testGt.astype(int)
    # drawGT(testGt, "test_gt.jpg")
    # drawGT(trainGt, "train_gt.jpg")
    return trainGt, testGt

select_gt = np.load('/home/aistudio/npys/t30uxv/select_gt.npy').astype(int)
select_rs = np.load('/home/aistudio/npys/t30uxv/select_rs.npy')
train_gt , test_gt = getTrainTest(select_gt, 0.05)

# 数据预处理：归一化
def normalize_data(select_rs):
    scaler = MinMaxScaler()
    H, W, C, T = select_rs.shape
    select_rs_reshaped = select_rs.reshape(-1, C)
    select_rs_normalized = scaler.fit_transform(select_rs_reshaped)
    select_rs_normalized = select_rs_normalized.reshape(H, W, C, T)
    return select_rs_normalized, scaler

def remap_labels(gt):
    # 将标签0映射为-1，其他标签减1
    remapped_gt = np.where(gt == 0, -1, gt - 1)
    
    # 计算映射后的唯一标签数量
    unique_labels = np.unique(remapped_gt)
    num_classes = len(unique_labels)
    
    return remapped_gt, num_classes-1


class RSDataset(Dataset):
    def __init__(self, rs_data, gt_data=None):
        # 假设rs_data的原始格式是H*W*C*T
        # 需要先将数据转换为T*C*H*W
        self.rs_data = rs_data.transpose((3, 2, 0, 1))  # 调整为T*C*H*W
        self.gt_data = gt_data
        self.H, self.W = rs_data.shape[:2]

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        row = idx // self.W
        col = idx % self.W
        # 对于每个像素点，获取其所有时间点的数据，现在格式为T*C
        sequence = self.rs_data[:, :, row, col].astype('float32')  # 获取T*C序列
        if self.gt_data is not None:
            label = self.gt_data[row, col]
            # 直接返回序列和标签，不对标签值进行特殊处理
            return sequence, label
        else:
            return sequence, np.array([row, col], dtype='float32')


class DCM(nn.Layer):
    def __init__(
        self, seed, input_feature_size, hidden_size, num_layers,
        bidirectional, dropout, num_classes
    ):
        super().__init__()
        self._set_reproducible(seed)

        # 修正：使用`dropout`参数而非`dropout_prob`
        self.lstm = nn.LSTM(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction='bidirectional' if bidirectional else 'forward',
            dropout=dropout if num_layers > 1 else 0,  # 当num_layers大于1时才设置dropout
            time_major=False
        )  # i/o: (batch, seq_len, num_directions*hidden_size)
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.fc = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=num_classes,
        )

    def _set_reproducible(self, seed, cudnn=False):
        paddle.seed(seed)
        np.random.seed(seed)
        if cudnn:
            paddle.set_flags({'FLAGS_cudnn_deterministic': True})

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), axis=1)
        fc_in = paddle.bmm(attn_weights.transpose([0, 2, 1]), lstm_out)
        fc_out = self.fc(fc_in)
        return fc_out.squeeze()


def calculate_metrics(true_labels, predicted_labels):
    """
    计算OA, AA, Kappa和mIoU
    """
    # 整体准确率 OA
    oa = accuracy_score(true_labels, predicted_labels)
    # Kappa 系数
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    # 混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # 每类的准确率
    acc_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    # 平均准确率 AA
    aa = np.mean(acc_per_class)
    # 每类的IoU
    iou_per_class = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - conf_matrix.diagonal())
    # 平均IoU mIoU
    miou = np.mean(iou_per_class)
    return oa, aa, kappa, miou, acc_per_class  # 确保返回每类的准确率

def evaluate_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    for data, label in data_loader:
        data = paddle.to_tensor(data)
        output = model(data)
        preds = output.argmax(axis=1).numpy()
        valid_idx = label != -1  # 有效索引，即非-1标签
        all_preds.extend(preds[valid_idx].flatten())
        all_labels.extend(label.numpy()[valid_idx].flatten())
    return calculate_metrics(all_labels, all_preds)


best_oa = -1  # 全局变量用于跟踪最佳OA

# 训练和评估模型，同时保存OA最高时的模型参数和指标
def train_and_evaluate(model, train_loader, test_loader, optimizer, loss_fn, epochs, save_path):
    global best_oa
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_id, (data, label) in enumerate(train_loader):
            data = paddle.to_tensor(data)
            label = paddle.to_tensor(label, dtype='int64')
            optimizer.clear_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.numpy()

        avg_train_loss = total_train_loss / len(train_loader)
        
        oa, aa, kappa, miou, _ = evaluate_metrics(model, test_loader)
        if oa > best_oa:
            best_oa = oa
            best_model_path = os.path.join(save_path, "model_param.pdparams")
            paddle.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}/{epochs}]: New best model saved at {best_model_path} with OA: {best_oa:.4f}")


def final_test_evaluation(model, test_loader, save_path):
    best_model_path = os.path.join(save_path, "model_param.pdparams")
    model_state_dict = paddle.load(best_model_path)
    model.set_state_dict(model_state_dict)
    model.eval()

    all_preds = []
    all_labels = []
    with paddle.no_grad():
        for data, label in test_loader:
            data = paddle.to_tensor(data)
            output = model(data)
            preds = output.argmax(axis=1).numpy()
            valid_idx = label.numpy() != -1
            all_preds.extend(preds[valid_idx].flatten())
            all_labels.extend(label.numpy()[valid_idx].flatten())

    oa, aa, kappa, miou, acc_per_class = calculate_metrics(all_labels, all_preds)
    np.save(os.path.join(save_path, "pre_label.npy"), np.array(all_preds))
    
    with open(os.path.join(save_path, "evaluation_metric.txt"), "w") as f:
        f.write(f"OA: {oa*100:.2f}%\n")
        f.write(f"AA: {aa*100:.2f}%\n")
        f.write(f"Kappa: {kappa*100:.2f}%\n")
        f.write(f"mIoU: {miou*100:.2f}%\n")
        for i, acc in enumerate(acc_per_class):
            f.write(f"class{i+1}: {acc*100:.2f}%\n")


# 数据集和训练比例的列表
datasets = ['t31tfj', 't31tfm', 't31tfm_1', 't32ulu']
train_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

for dataset in datasets:
    select_gt_path = f'/home/aistudio/npys/{dataset}/select_gt.npy'
    select_rs_path = f'/home/aistudio/npys/{dataset}/select_rs.npy'
    select_gt = np.load(select_gt_path).astype(int)
    select_rs = np.load(select_rs_path)

    for train_ratio in train_ratios:
        print(f"Processing dataset {dataset} with train ratio {train_ratio}")
        train_gt, test_gt = getTrainTest(select_gt, train_ratio)
        select_rs_normalized, _ = normalize_data(select_rs)
        train_gt_remapped, num_classes = remap_labels(train_gt)
        test_gt_remapped, _ = remap_labels(test_gt)

        train_dataset = RSDataset(select_rs_normalized, train_gt_remapped)
        test_dataset = RSDataset(select_rs_normalized, test_gt_remapped)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = DCM(seed=313, input_feature_size=select_rs_normalized.shape[2], hidden_size=256, num_layers=2, bidirectional=True, dropout=0.5, num_classes=num_classes)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
        loss_fn = paddle.nn.CrossEntropyLoss(ignore_index=-1)

        save_path = f'/home/aistudio/DCM/{dataset}/{train_ratio}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        best_oa = -1  # 重置最佳OA
        train_and_evaluate(model, train_loader, test_loader, optimizer, loss_fn, 200, save_path)
        final_test_evaluation(model, test_loader, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_gt_path', type=str, required=True, help='路径到 select_gt.npy 文件')
    parser.add_argument('--select_rs_path', type=str, required=True, help='路径到 select_rs.npy 文件')
    parser.add_argument('--train_ratio', type=float, required=True, help='训练集的比例')
    parser.add_argument('--epochs', type=int, default=200, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--save_path', type=str, required=True, help='模型和指标保存的路径')

    args = parser.parse_args()
    main(args)