import copy
import os
import random
from collections import Counter

import scipy.io
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
from importlib import import_module
import argparse
import logging
import sys
import pandas as pd
import math

# ---------------------------------- Parameters ----------------------------------

parser = argparse.ArgumentParser("ours")
parser.add_argument("--dataset", type=str, default="France", choices=['France', 'German', 'Pastis'])
parser.add_argument("--is_analysis", action="store_true", help="Enable analysis mode")
parser.add_argument("--period", type=int, default=1)
parser.add_argument("--compressor", type=str, default="gzip", choices=["gzip", "bz2", "zstandard", "lzma"])
parser.add_argument("--seed", type=int, default=32, choices=[32, 47, 2024, 400, 21])
parser.add_argument("--concat_mode", type=str, default="bp_pb", choices=["bp", "pb", "bp_pb", "pb_bp"])
parser.add_argument("--code", type=str, default="char", choices=["num", "char"])
parser.add_argument("--alphabet_len", type=int, default=22)
parser.add_argument("--mapping", type=str, default="equal_interval", choices=["equal_interval", "equal_quantile"])
parser.add_argument("--str_code", type=str, default="normal", choices=["gray", "normal"])
parser.add_argument("--train_num", type=float, default=0.2, choices=[0.2, 0.1, 50, 20, 10, 5])
parser.add_argument("--single_ncd", action='store_true')
parser.add_argument("--k", type=int, default=2)


args = parser.parse_args()


def equal_quantile(array, elements_per_part):
    bins = []
    for i in range(0, len(array), elements_per_part):
        start = array[i]
        end = array[min(i + elements_per_part, len(array)) - 1]
        bins.append((start, end))
    return bins


def equal_interval(array, letter_num):
    bins = []
    max_num = np.max(array)
    min_num = np.min(array)
    interval = 1.0 * (max_num - min_num) / letter_num
    for i in range(1, letter_num):
        bins.append((min_num, min_num + interval))
        min_num += interval
    bins.append((min_num, max_num))
    return bins


def generate_gray_codes(n):
    if n == 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]

    # 递归生成前一位的格雷码
    prev_codes = generate_gray_codes(n - 1)
    result = []

    # 添加“0”到前一位的格雷码
    for code in prev_codes:
        result.append("0" + code)

    # 添加“1”到前一位的格雷码的镜像
    for code in reversed(prev_codes):
        result.append("1" + code)

    return result


def alphabet(data, alphabet_len, mapping, str_mode):
    rate = float(1 / alphabet_len)   # 依据字母表长度，来确定等概率是多少
    sorted_data = np.sort(data.reshape(-1))
    part_size = int(len(sorted_data) * rate)

    if mapping == "equal_interval":  # 等间距划分
        bins = equal_interval(sorted_data, alphabet_len)
    else:  # 等概率划分
        bins = equal_quantile(sorted_data, part_size)

    # 构建映射表
    mapping_table = {}

    if str_mode == "normal":

        for i, (start, end) in enumerate(bins):
            letter_index = i % 52
            if letter_index < 26:
                letter = chr(ord('a') + letter_index)
            else:
                letter = chr(ord('A') + letter_index - 26)
            mapping_table[(start, end)] = letter

    elif str_mode == "gray":

        bits_needed = math.ceil(math.log2(len(bins)))
        gray_codes = generate_gray_codes(bits_needed)
        for i, (start, end) in enumerate(bins):
            mapping_table[(start, end)] = gray_codes[i]

    return mapping_table


def merge_columns_by_bands(array, columns_to_merge, sep):
    result = []
    for row in array:
        merged_row = [sep.join(map(str, row[i:i+columns_to_merge])) for i in range(0, len(row), columns_to_merge)]
        result.append(merged_row)
    return np.array(result)


def merge_all_columns(data):
    return np.array([' '.join(map(str, row)) for row in data])


# def add_space(s):
#     return ' '.join([s[i:i+args.period] for i in range(0, len(s), args.period)])

def ncd(c1, c2, c12):
    return (c12 - min(c1, c2)) / max(c1, c2)


def clen(data, comp):
    return len(comp.compress(data))


def split(dataset, train_gt, test_gt):
    train_gt = train_gt.flatten()
    test_gt = test_gt.flatten()

    train_indices = np.nonzero(train_gt)[0]
    test_indices = np.nonzero(test_gt)[0]

    x_train = dataset[train_indices]
    y_train = train_gt[train_indices].astype(int).astype(str)

    x_test = dataset[test_indices]
    y_test = test_gt[test_indices].astype(int).astype(str)

    train_set = np.hstack((x_train, y_train.reshape(-1, 1)))
    test_set = np.hstack((x_test, y_test.reshape(-1, 1)))
    return train_set, test_set, train_indices, test_indices


def compute_precomputed_lengths(data, comp):
    flat_data = data.ravel()

    # Compress each element using a list comprehension and convert to bytes
    tmp = [clen(element.encode(), comp) for element in flat_data]

    # Reshape the compressed data back to the original array shape
    precomputed_lengths = np.array(tmp).reshape(data.shape)

    return precomputed_lengths


def break_ties(ncds_x1, train, k):
    top_k = np.argsort(ncds_x1)[:k]
    top_k_labels = train[top_k, -1]
    unique_labels, label_counts = np.unique(top_k_labels, return_counts=True)
    max_count = np.max(label_counts)
    most_common_labels = unique_labels[label_counts == max_count]
    if len(most_common_labels) == 1:
        predict_label = most_common_labels[0]
    else:
        mask = np.isin(top_k_labels, most_common_labels)
        matching_indices = np.where(mask)[0]
        predict_label = top_k_labels[matching_indices[0]]
    return predict_label


def gzip_knn(tes, train, precomputed_lengths, compressor, single_ncd, index):
    comp = import_module(compressor)
    if compressor == "zstandard":
        comp = comp.ZstdCompressor()

    ncds_x1 = np.zeros(len(train))
    x1, x1_label = tes[:-1], tes[-1]
    result = [x1_label]

    if single_ncd:

        x1 = ' '.join(x1)
        l1 = clen(x1.encode(), comp)

        for j, tra in enumerate(train):
            x2, x2_label = tra[:-1], tra[-1]
            x2 = ' '.join(x2)
            l2 = precomputed_lengths[j]
            l12 = clen(' '.join([x1, x2]).encode(), comp)
            ncds_x1[j] = ncd(l1, l2, l12)

    else:

        l1 = np.array([clen(x.encode(), comp) for x in x1])

        for j, tra in enumerate(train):
            tmp = 0
            x2, x2_label = tra[:-1], tra[-1]
            for i in range(len(x1)):
                l2 = precomputed_lengths[j][i]
                l12 = clen(' '.join([x1[i], x2[i]]).encode(), comp)
                tmp += ncd(l1[i], l2, l12)
            ncds_x1[j] = tmp/len(x1)

    # the following code is to break ties
    # loop through k from 1 to 18, and 21, 35, 51
    for i in range(1, 19):
        result.append(break_ties(ncds_x1, train, i))
    result.append(break_ties(ncds_x1, train, 21))
    result.append(break_ties(ncds_x1, train, 35))
    result.append(break_ties(ncds_x1, train, 51))

    return result, index


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


def calculate_metrics(result):
    true_labels = result[:, 0]
    predicted_labels = result[:, 1]
    OA = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    mIoU = calculate_mIoU(cm)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    AA = np.mean(class_acc)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return OA, mIoU, AA, class_acc, kappa


def getTrainTest_1(label, trainNum, seed):
    random.seed(seed)
    testGt = copy.deepcopy(label)
    trainGt = np.zeros(label.shape)
    labelNum = int(max(label.ravel())) + 1
    for lab in range(1, labelNum):
        labIndex = np.where(label == lab)
        if trainNum < 1:
            train_num = 1 if int(len(labIndex[0]) * trainNum) == 0 and len(labIndex[0]) != 0 else int(
                len(labIndex[0]) * trainNum)
        else:
            train_num = int(trainNum) if len(labIndex[0]) != 0 else 0
        randomList = random.sample(range(0, len(labIndex[0])), train_num)
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
    testArray = testGt.ravel()
    print(Counter(testArray))
    return trainGt, testGt


def getTrainTest_2(label, trainNum, trainSeed=32, testSeed=32):
    tempGt = copy.deepcopy(label)
    trainGt = np.zeros(label.shape)
    testGt = np.zeros(label.shape)
    labelNum = int(max(label.ravel())) + 1
    for lab in range(1, labelNum):
        labIndex = np.where(label == lab)
        if trainNum < 1:
            train_num = 1 if int(len(labIndex[0]) * trainNum) == 0 and len(labIndex[0]) != 0 else int(len(labIndex[0]) * trainNum)
        else:
            train_num = int(trainNum) if len(labIndex[0]) != 0 else 0
        random.seed(trainSeed)
        randomList = random.sample(range(0, len(labIndex[0])), train_num)
        for randInt in randomList:
            x = labIndex[0][randInt]
            y = labIndex[1][randInt]
            trainGt[x][y] = lab
            tempGt[x][y] = 0

        labIndex = np.where(tempGt == lab)
        test_num = train_num
        random.seed(testSeed)
        randomList = random.sample(range(0, len(labIndex[0])), test_num)
        for randInt in randomList:
            x = labIndex[0][randInt]
            y = labIndex[1][randInt]
            testGt[x][y] = lab
    trainGt = trainGt.astype(int)
    testGt = testGt.astype(int)
    trainArray = trainGt.ravel()
    testArray = testGt.ravel()
    return trainGt, testGt


def generate_filename(args):
    param_str = '_'.join(f"{key}={value}" for key, value in vars(args).items() if value is not None)
    return f"{param_str}"


if __name__ == '__main__':
    # ---------------------------------- Load Data ----------------------------------
    data_path = f'{args.dataset}/rs.npy'
    gt_path = f'{args.dataset}/gt.npy'
    data = np.load(data_path)
    gt = np.load(gt_path).astype(int)
    if args.is_analysis:
        train_gt, test_gt = getTrainTest_2(gt, args.train_num, args.seed, args.seed)
    else:
        train_gt, test_gt = getTrainTest_1(gt, args.train_num, args.seed)
    print(f"Data shape: {data.shape}")

    comp = import_module(args.compressor)
    if args.compressor == "zstandard":
        comp = comp.ZstdCompressor()

    save_name = generate_filename(args)

    log_format = '%(asctime)s %(message)s'
    log_path = f'{save_name}.txt'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args_dict = vars(args)
    # Iterate over the dictionary and print each parameter and its value
    for param, value in args_dict.items():
        print(f"{param}: {value}")

    # ---------------------------------- Concatenate Data ----------------------------------
    # p: period, b: bands
    data = np.transpose(data, (0, 1, 3, 2))  # (w, h, p, b)
    bp_slices = np.transpose(data, (0, 1, 3, 2))[:, :, :, :args.period]
    pb_slices = data[:, :, :args.period, :]
    w, h, b, p = bp_slices.shape

    if args.concat_mode == "bp":
        reshaped_data = bp_slices.reshape(w * h, b * p)
    elif args.concat_mode == "pb":
        reshaped_data = pb_slices.reshape(w * h, p * b)
    elif args.concat_mode == "bp_pb":
        reshaped_data = np.concatenate((bp_slices.reshape(w * h, b * p), pb_slices.reshape(w * h, p * b)), axis=1)
    elif args.concat_mode == "pb_bp":
        reshaped_data = np.concatenate((pb_slices.reshape(w * h, p * b), bp_slices.reshape(w * h, b * p)), axis=1)
    else:
        raise ValueError(f"Invalid concat mode: {args.concat_mode}")
    print(f"Reshaped data shape: {reshaped_data.shape}")

    # ---------------------------------- Merge Data ----------------------------------
    if args.code == "num":

        final_data = merge_columns_by_bands(reshaped_data, columns_to_merge=p, sep=' ')
        print(f"Final data shape: {final_data.shape}")
        # logging.info(f"Final data: {final_data}")

    elif args.code == "char":

        mapping_table = alphabet(reshaped_data, args.alphabet_len, args.mapping, args.str_code)

        def map_value_to_letter(value):  
            for (lower_bound, upper_bound), letter in mapping_table.items():
                if lower_bound <= value < upper_bound:
                    return letter
            return None

        vectorized_mapping = np.vectorize(map_value_to_letter)
        tmp = vectorized_mapping(reshaped_data)
        print(f"Mapping table: {mapping_table}")
        print(f"Tmp data shape: {tmp.shape}")
        # logging.info(f"Tmp data: {tmp}")

        if args.concat_mode == "bp":
            final_data = merge_columns_by_bands(tmp, columns_to_merge=p, sep='')
        elif args.concat_mode == "pb":
            final_data = merge_columns_by_bands(tmp, columns_to_merge=b, sep='')
        elif args.concat_mode == "bp_pb":
            bp_part = merge_columns_by_bands(tmp[:, :b*p], columns_to_merge=p, sep='')
            pb_part = merge_columns_by_bands(tmp[:, b*p:], columns_to_merge=b, sep='')
            final_data = np.concatenate((bp_part, pb_part), axis=1)
        elif args.concat_mode == "pb_bp":
            pb_part = merge_columns_by_bands(tmp[:, :p*b], columns_to_merge=b, sep='')
            bp_part = merge_columns_by_bands(tmp[:, p*b:], columns_to_merge=p, sep='')
            final_data = np.concatenate((pb_part, bp_part), axis=1)

        print(f"Final data shape: {final_data.shape}")
        print(f"Final data: {final_data}")

    # ---------------------------------- Split Data ----------------------------------

    train, test, train_indices, test_indices = split(final_data, train_gt, test_gt)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    # logging.info(f"Train: {train}")
    # logging.info(f"Test: {test}")

    # ---------------------------------- Precompute Lengths ----------------------------------
    if args.single_ncd:

        train_label = train[:, -1].reshape(-1, 1)
        train = merge_all_columns(train[:, :-1]).reshape(-1, 1)
        train = np.concatenate((train, train_label), axis=1)
        test_label = test[:, -1].reshape(-1, 1)
        test = merge_all_columns(test[:, :-1]).reshape(-1, 1)
        test = np.concatenate((test, test_label), axis=1)
        print(f"2Train shape: {train.shape}")
        print(f"2Test shape: {test.shape}")
        # logging.info(f"Train: {train}")
        # logging.info(f"Test: {test}")

        precomputed_lengths = np.array([clen(row[0].encode(), comp) for row in train])
        print(f"Precomputed lengths shape: {precomputed_lengths.shape}")
        print(f"Precomputed lengths: {precomputed_lengths}")

    else:

        precomputed_lengths = compute_precomputed_lengths(train[:, :-1], comp)
        print(f"Precomputed lengths shape: {precomputed_lengths.shape}")
        print(f"Precomputed lengths: {precomputed_lengths}")

    # ---------------------------------- Compressor with KNN ----------------------------------
    results = [[] for _ in range(len(test))]
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, tes in enumerate(test):
            future = executor.submit(
                gzip_knn,
                tes,
                train,
                precomputed_lengths,
                args.compressor,  # cannot use 'comp' here, because it is not pickleable
                args.single_ncd,
                i,
            )
            futures.append(future)

        progress = tqdm(as_completed(futures), total=len(futures), desc="Processing")

        for future in progress:
            try:
                result, index = future.result()
                results[index] = result
            except Exception as e:
                logging.error(f"Error: {e}")

    # ---------------------------------- Calculate Metrics ----------------------------------
    results = np.array(results)
    print(f"Results shape: {results.shape}")
    print(f"Results: {results}")
    label = results[:, 0]
    row_list = []
    for i in range(1, results.shape[1]):
        pred_label = results[:, i]
        result_k = np.hstack((label.reshape(-1, 1), pred_label.reshape(-1, 1)))
        OA, mIoU, AA, class_accuracy, kappa = calculate_metrics(result_k)
        print(f"{i}, OA: {OA}\nmIoU: {mIoU}\nAA: {AA}\nclass accuracy: {class_accuracy}\nkappa: {kappa}\n")
        row_list.append((i, OA, mIoU, AA, kappa, *class_accuracy))

    columns = ['k', 'OA', 'mIoU', 'AA', 'kappa'] + [f'class{i}_acc' for i in range(len(class_accuracy))]
    df = pd.DataFrame(row_list, columns=columns)
    df.to_excel(f'{save_name}.xlsx', index=False)

    # ---------------------------------- Draw Classification map ----------------------------------
    pred_map = np.zeros(gt.shape)
    iterator = 0
    for index in test_indices:
        x = index // gt.shape[1]
        y = index % gt.shape[1]
        pred_map[x][y] = results[iterator, args.k]
        iterator += 1

    for index in train_indices:
        x = index // gt.shape[1]
        y = index % gt.shape[1]
        pred_map[x][y] = train_gt[x][y]

    pred_map = pred_map.astype(int)
    np.save(f'{save_name}.npy', pred_map)







