# low-resource crop classification from multi-spectral time series using lossless compressors

## Dataset
Since the dataset is too large, you can download the data we use from the following links of baiduyun [Dataset](https://pan.baidu.com/s/1zYGEa1OOLbVkexjC1FAwSQ?pwd=4dbe) (access code: 4dbe)

## Usage
### Dependencies
~~~
pip install einops==0.4.0
pip install patool==1.12
pip install reformer-pytorch==1.4.4
pip install sktime==0.16.1
pip install sympy==1.11.1
~~~
### Ours
As an example, use the following command to run our method on four parcel.
~~~
cd model/

python ours_admin.py
~~~

### Deep learning models
#### TempCNN, LSTM and InceptionTime
~~~
cd model/TempCNN_LSTM_InceptionTime/

python tempcnn.py \
--areas t30uxv \
--train_num 0.2 \
--period 43 \
--model "Choose the model from TempCNN, LSTM, or InceptionTime."
~~~

#### TimesNet, LightTS, DLinear, iTransformer, Informer and Transformer
~~~
cd model/Time-series-lib/

python train.py \
--foldname "/path/to/dataset" \
--train_ratio 0.5 0.2 0.1 0.05 0.02 0.01 \
--periods 1 2 3 4 \
--patch_names t30uxv t31tfj t31tfm t32ulu \
--data_paths CropMapping_t30uxv CropMapping_t31tfj CropMapping_t31tfm CropMapping_t32ulu \
--models "Choose the model from TimesNet LightTS DLinear Transformer Informer iTransformer."
~~~

#### DCM
~~~
cd model/DCM/

python DCM.py \
--select_gt_path "/path/to/select_gt.npy" \
--select_rs_path "/path/to/select_rs.npy" \
--train_ratio 0.05 \
--epochs 200 \
--batch_size 32 \
--learning_rate 0.001 \
--save_path "/path/to/save/model"
~~~

## Contact
If you have any questions or suggestions, feel free to contact:

feifanzhang@cau.edu.cn

Or describe it in Issues.


## Acknowledgement

All the experiment datasets are public, and we obtain them from the following links:
- [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

The deep learning models implementations from this repository are based on the following github repositories:
- TempCNN, LSTM and InceptionTime from [BreizhCrops](https://github.com/dl4sits/BreizhCrops)
- TimesNet, LightTS, DLinear, iTransformer, Informer, Transformer from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- DCM from [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping)

