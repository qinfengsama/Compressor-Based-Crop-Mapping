import argparse
import numpy as np
import subprocess
import copy
import random
from collections import Counter


def getTrainTest(label, trainNum):
    # 设置随机种子
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


def toTS(GT, RS, tsFilePath, period=1):
    RS3D = RS.reshape((RS.shape[0] * RS.shape[1], RS.shape[2], RS.shape[3]))
    if period == 4:
        RS3D = RS3D[:, :, 0:RS3D.shape[2]]
    else:
        RS3D = RS3D[:, :, 0:RS3D.shape[2] // 4 * period]
    GT1D = GT.ravel()
    classLabel = np.unique(GT1D)
    strClassLabel = ""
    for la in classLabel:
        strClassLabel += " " + str(la)
    with open(tsFilePath, 'w') as file:
        file.write("@problemName CropMapping_" + patch_name + "\n")
        file.write("@timeStamps false\n")
        file.write("@missing false\n")
        file.write("@univariate false\n")
        file.write("@dimensions" + str(RS3D.shape[2]) + "\n")
        file.write("@equalLength true\n")
        file.write("@seriesLength" + str(RS3D.shape[1]) + "\n")
        file.write("@classLabel true" + strClassLabel + "\n")
        file.write("@data\n")
        for index, label in enumerate(GT1D):
            if label != 0:
                data = RS3D[index]
                writeData = ""
                for time in range(data.shape[1]):
                    for band in range(data.shape[0]):
                        writeData += str(data[band, time])
                        if band != data.shape[0] - 1:
                            writeData += ","
                        else:
                            writeData += ":"
                writeData += str(label)
                writeData += '\n'
                file.write(writeData)


def main(foldname, trainNums, periods, patch_names, data_paths, models):
    for model in models:
        for trainNum in trainNums:
            for period in periods:
                for patch_name in patch_names:
                    # Load ground truth and remote sensing data
                    gt = np.load(f"{foldname}/{patch_name}/select_gt.npy")
                    rs = np.load(f"{foldname}/{patch_name}/select_rs.npy")
                    # Split ground truth into training and testing sets
                    train_gt, test_gt = getTrainTest(gt, trainNum)
                    # Convert training and testing data to time series format
                    toTS(train_gt, rs, f"{foldname}/CropMapping_{patch_name}/CropMapping_{patch_name}_TRAIN.ts", period)
                    toTS(test_gt, rs, f"{foldname}/CropMapping_{patch_name}/CropMapping_{patch_name}_TEST.ts", period)

                for data_path in data_paths:
                    # Build the command line argument list
                    command = [
                        'python', '-u', f"{foldname}/run.py",
                        '--task_name', 'classification',
                        '--is_training', '1',
                        '--root_path', f"{foldname}/{data_path}/",
                        '--model_id', f"{data_path}_{period}in4Period_{trainNum}train",
                        '--model', model,
                        '--data', 'UEA',
                        '--e_layers', '3',
                        '--batch_size', '32',
                        '--d_model', '128',
                        '--d_ff', '256',
                        '--top_k', '3',
                        '--des', 'Exp',
                        '--itr', '1',
                        '--learning_rate', '0.001',
                        '--train_epochs', '200',
                        '--patience', '10',
                        '--enc_in', '3'
                    ]

                    # Run the command
                    subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run time series model training and testing')
    parser.add_argument('--foldname', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--train_ratio', type=float, nargs='+', default=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
                        help='List of training set ratios')
    parser.add_argument('--periods', type=int, nargs='+', default=[1, 2, 3, 4], help='List of periods')
    parser.add_argument('--patch_names', type=str, nargs='+', required=True, help='List of patch names')
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, help='List of data paths')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models')

    args = parser.parse_args()

    main(args.foldname, args.train_ratio, args.periods, args.patch_names, args.data_paths, args.models)