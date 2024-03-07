import torch
import argparse
import pandas as pd
import numpy as np
from breizhcrops.models import LSTM, TempCNN, MSResNet, TransformerModel, StarRNN, OmniScaleCNN, \
    PETransformerModel
from InceptionTime import InceptionTime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import copy
from collections import Counter
import logging
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

parser = argparse.ArgumentParser(description='Train a model on the Pastis dataset.')

parser.add_argument('--areas', type=str, default='t30uxv', help='area name')
parser.add_argument('--train_num', type=float, default=0.5, help='train number')
parser.add_argument('--period', type=int, default=43, help='period name')

parser.add_argument(
    '--model', type=str, default="LSTM",
    choices=["LSTM", "StarRNN", "InceptionTime", "MSResNet", "TransformerEncoder", "TempCNN", "OmniScaleCNN"], )
parser.add_argument(
    '--batchsize', type=int, default=256, help='batch size (number of time series processed simultaneously)')
parser.add_argument(
    '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
parser.add_argument(
    '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                             'default will check by torch.cuda.is_available() ')
parser.add_argument(
    '--preload-ram', action='store_true', help='load dataset into RAM upon initialization')

args = parser.parse_args()

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


class Pastis(Dataset):
    def __init__(self, x, y, indices):
        self.x = x
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index]).type(torch.FloatTensor)
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        return x, y, self.indices[index]


def get_model(modelname, ndims, num_classes, sequencelength, device, **hyperparameter):
    modelname = modelname.lower()  # make case invariant
    if modelname == "omniscalecnn":
        model = OmniScaleCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength,
                             **hyperparameter).to(device)
    elif modelname == "lstm":
        model = LSTM(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname == "starrnn":
        model = StarRNN(input_dim=ndims,
                        num_classes=num_classes,
                        bidirectional=False,
                        use_batchnorm=False,
                        use_layernorm=True,
                        device=device,
                        **hyperparameter).to(device)
    elif modelname == "inceptiontime":
        model = InceptionTime(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device)
    elif modelname == "msresnet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname in ["transformerencoder", "transformer"]:
        model = TransformerModel(input_dim=ndims, num_classes=num_classes,
                                 activation="relu",
                                 **hyperparameter).to(device)
    elif modelname in ["petransformer"]:
        model = PETransformerModel(input_dim=ndims, num_classes=num_classes,
                                   activation="relu",
                                   **hyperparameter).to(device)
    elif modelname == "tempcnn":
        model = TempCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(
            device)
    else:
        raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")

    return model


def getTrainTest(label, trainNum):
    random.seed(32)
    testGt = copy.deepcopy(label)
    trainGt = np.zeros(label.shape)
    labelNum = int(max(label.ravel())) + 1
    for lab in range(1, labelNum):
        labIndex = np.where(label == lab)
        randomList = random.sample(range(0, len(labIndex[0])), int(len(labIndex[0]) * trainNum))
        for randInt in randomList:
            x = labIndex[0][randInt]
            y = labIndex[1][randInt]
            trainGt[x][y] = lab
            testGt[x][y] = 0
    trainGt = trainGt.astype(int)
    testGt = testGt.astype(int)
    # drawGT(testGt, "test_gt.jpg")
    # drawGT(trainGt, "train_gt.jpg")
    trainArray = trainGt.ravel()
    print(Counter(trainArray))
    testArray = testGt.ravel()
    print(Counter(testArray))
    return trainGt, testGt


def calculate_mIoU(confusion_matrix):
    # Intersection is the diagonal of the confusion matrix
    intersection = np.diag(confusion_matrix)
    # Union is the sum over ground truth and prediction minus the intersection
    ground_truth_set = np.sum(confusion_matrix, axis=1)
    predicted_set = np.sum(confusion_matrix, axis=0)
    union = ground_truth_set + predicted_set - intersection

    # IoU for each class, and avoiding division by 0
    iou = intersection / (union + 1e-6)
    # Mean IoU across all classes
    mIoU = np.nanmean(iou)  # Using nanmean to ignore NaN values in case of division by 0
    return mIoU


def calculate_metrics(true_labels, predicted_labels):
    OA = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    mIoU = calculate_mIoU(cm)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    AA = np.mean(class_acc)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return OA, mIoU, AA, class_acc, kappa


def split(dataset, train_gt, test_gt):
    h, w, p, b = dataset.shape
    dataset = dataset.reshape(h * w, p, b)
    train_gt = train_gt.flatten()
    test_gt = test_gt.flatten()

    train_indices = np.nonzero(train_gt != -1)[0]
    test_indices = np.nonzero(test_gt != -1)[0]

    x_train = dataset[train_indices]  # (n, p, b)
    y_train = train_gt[train_indices]

    x_test = dataset[test_indices]
    y_test = test_gt[test_indices]

    return x_train, y_train, x_test, y_test, train_indices, test_indices


def get_dataloader(data, train_gt, test_gt, batchsize, workers, preload_ram=False):
    x_train, y_train, x_test, y_test, train_indices, test_indices = split(data, train_gt, test_gt)

    train = Pastis(x_train, y_train, train_indices)
    test = Pastis(x_test, y_test, test_indices)
    train_loader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=workers)

    return train_loader, test_loader, train_indices, test_indices

    # if mode == "unittest":
    #     belle_ile = breizhcrops.BreizhCrops(region="belle-ile", root=datapath)
    # else:
    #     frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath,
    #                                     preload_ram=preload_ram, level=level)
    #     frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,
    #                                     preload_ram=preload_ram, level=level)
    #     frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath,
    #                                     preload_ram=preload_ram, level=level)
    #
    # if "evaluation" in mode:
    #         frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath,
    #                                         preload_ram=preload_ram, level=level)
    #
    # if mode == "evaluation" or mode == "evaluation1":
    #     traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
    #     testdataset = frh04
    # elif mode == "evaluation2":
    #     traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh04])
    #     testdataset = frh03
    # elif mode == "evaluation3":
    #     traindatasets = torch.utils.data.ConcatDataset([frh01, frh03, frh04])
    #     testdataset = frh02
    # elif mode == "evaluation4":
    #     traindatasets = torch.utils.data.ConcatDataset([frh02, frh03, frh04])
    #     testdataset = frh01
    # elif mode == "validation":
    #     traindatasets = torch.utils.data.ConcatDataset([frh01, frh02])
    #     testdataset = frh03
    # elif mode == "unittest":
    #     traindatasets = belle_ile
    #     testdataset = belle_ile
    # else:
    #     raise ValueError("only --mode 'validation' or 'evaluation' allowed")


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true, _ = batch
            loss = criterion(model.forward(x.to(device)), y_true.to(device))
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)


def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, field_id = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(
            field_ids_list)


def main(args):
    data_path = f'pastis/{args.areas}/select_rs.npy'
    gt_path = f'pastis/{args.areas}/select_gt.npy'

    data = np.transpose(np.load(data_path), (0, 1, 3, 2))[:, :, :args.period, :]  # (h, w, p, b)
    h, w, sequencelength, ndims = data.shape
    logging.info(f"Loaded data with shape {data.shape}")

    gt = np.load(gt_path).astype(int)
    num_classes = len(np.unique(gt)) - 1  # 0 is undefined label
    train_gt, test_gt = getTrainTest(gt, args.train_num)

    train_gt = train_gt - 1  # -1 is undefined label
    test_gt = test_gt - 1

    train_loader, test_loader, train_indices, test_indices = get_dataloader(data=data, train_gt=train_gt,
                                                                            test_gt=test_gt, batchsize=args.batchsize,
                                                                            workers=args.workers,
                                                                            preload_ram=args.preload_ram)

    epochs, learning_rate, weight_decay = select_hyperparameter(args.model)

    save_name = f"{args.areas}_{args.model}_{str(args.period)}_{str(args.train_num)}_{str(epochs)}_{str(learning_rate)}_{str(weight_decay)}"
    log_format = '%(asctime)s %(message)s'
    log_path = f'{save_name}.txt'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, sequencelength, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.modelname += f"_learning-rate={learning_rate}_weight-decay={weight_decay}"
    logging.info(f"Initialized {model.modelname}")
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(epochs):
        logging.info(f"train epoch {epoch}")
        train_epoch(model, optimizer, criterion, train_loader, device)
    losses, y_true, y_pred, y_score, field_ids = test_epoch(model, criterion, dataloader=test_loader, device=device)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    row_list = []
    OA, mIoU, AA, class_acc, kappa = calculate_metrics(y_true, y_pred)
    row_list.append((OA, mIoU, AA, kappa, *class_acc))
    columns = ['OA', 'mIoU', 'AA', 'kappa'] + [f'class{i}_acc' for i in range(len(class_acc))]
    df = pd.DataFrame(row_list, columns=columns)
    df.to_excel(f'{save_name}.xlsx', index=False)
    logging.info(f"OA={OA}, mIoU={mIoU}, AA={AA}, kappa={kappa}")
    logging.info(f"class_acc={class_acc}")

    # ---------------------------------- Predict Map ----------------------------------
    pred_map = np.zeros(gt.shape) - 1  # -1 is undefined label
    iterator = 0
    for index in test_indices:
        x = index // gt.shape[1]
        y = index % gt.shape[1]
        pred_map[x][y] = y_pred[iterator]
        iterator += 1

    iterator = 0
    for index in train_indices:
        x = index // gt.shape[1]
        y = index % gt.shape[1]
        pred_map[x][y] = train_gt[x][y]
        iterator += 1
    pred_map = np.array(pred_map)
    pred_map = pred_map.astype(int) + 1  # 0 is undefined label
    np.save(f'{save_name}.npy', pred_map)

    model_state = model.state_dict()
    torch.save(dict(model_state=model_state), save_name + ".pth")


def select_hyperparameter(model):
    """
    a function to select training-specific hyperparameter. the model-specific hyperparameter should be set
    in the defaults of the respective model parameters.
    """
    if model == "LSTM":
        epochs, learning_rate, weight_decay = 17, 0.009880117756170353, 5.256755602421856e-07
    elif model == "StarRNN":
        epochs, learning_rate, weight_decay = 17, 0.008960989762612663, 2.2171861339535254e-06
    elif model == "InceptionTime":
        epochs, learning_rate, weight_decay = 23, 0.0005930998594456241, 1.8660112778851542e-05
    elif model == "MSResNet":
        epochs, learning_rate, weight_decay = 23, 0.0006271686393146093, 4.750234747127917e-06
    elif model == "TransformerEncoder":
        epochs, learning_rate, weight_decay = 30, 0.00017369201853408445, 3.5156458637523697e-06
    elif model == "TempCNN":
        epochs, learning_rate, weight_decay = 11, 0.00023892874563871753, 5.181869707846283e-05
    elif model == "OmniScaleCNN":
        epochs, learning_rate, weight_decay = 19, 0.001057192239267413, 2.2522895556530792e-07
    return epochs, learning_rate, weight_decay


if __name__ == '__main__':
    main(args)