# A Pixel Can Be Compressed as a Spectral-Temporal Text
## Citation
If you re-use this work, please cite:

## Dataset
The original dataset originates from [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

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
As an example, use the following command to run our method on the T30UXV parcel.
~~~
cd model/ours/

python 0227_tsc_gzip.py \
--dataset pastis \
--area t30uxv \
--period 43 \
--compressor gzip \
--concat_mode bp_pb \
--code char \
--alphabet_len 51 \
--mapping equal_interval \
--str_code normal \
--train_num 0.5 \
--k 8
~~~

### Deep learning models
We also provide our pre-trained models to reproduce the experimental results in [Pretrained_Models](https://pan.baidu.com/s/1SB0ylQDYVwCKVWQt19qeaA) (access code: v1uv)
#### TempCNN, LSTM and InceptionTime
~~~
cd model/TempCNN_LSTM_InceptionTime/

python tempcnn.py \
--areas t30uxv \
--train_num 0.5 \
--period 43 \
--model "Choose the model from TempCNN, LSTM, or InceptionTime."
~~~

#### TimesNet, LightTS, DLinear, iTransformer, Informer and Transformer
~~~
cd model/Time-series-lib/

python train.py \
--foldname "/path/to/dataset" \
--train_ratio 0.5 0.2 \
--periods 1 2 \
--patch_names t30uxv t32ulu \
--data_paths CropMapping_t30uxv CropMapping_t31tfj \
--models TimesNet LightTS "Choose the model from TimesNet LightTS DLinear Transformer Informer iTransformer."
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


## Acknowledgement
The deep learning models implementations from this repository are based on the following github repositories.
- TempCNN, LSTM and InceptionTime from [BreizhCrops](https://github.com/dl4sits/BreizhCrops)
- TimesNet, LightTS, DLinear, iTransformer, Informer, Transformer from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- DCM from [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping)

