# A Pixel Can Be Compressed as a Spectral-Temporal Text
## Citation
If you re-use this work, please cite:

## Dataset
The original dataset originates from [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

Since the dataset is too large, you can download the data we use from the following links of baiduyun [Baidu Netdisk](https://pan.baidu.com/s/1zYGEa1OOLbVkexjC1FAwSQ?pwd=4dbe) (access code: 4dbe)

## Usage
### Dependencies
~~~
einops==0.4.0
patool==1.12
reformer-pytorch==1.4.4
sktime==0.16.1
sympy==1.11.1
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
We also provide our pretrained models for inference in [Baidu Netdisk](https://pan.baidu.com/s/1SB0ylQDYVwCKVWQt19qeaA) (access code: v1uv)
#### TempCNN, LSTM and InceptionTime
~~~
cd model/TempCNN_LSTM_InceptionTime/

python tempcnn.py \
--areas t30uxv \
--train_num 0.5 \
--period 43 \
--model "Choose the model from TempCNN, LSTM, or InceptionTime."
~~~

#### TimesNet, LightTS, DLinear, iTransformer, Informer, Transformer
As an example, use the following command to run these methods for all experients.
~~~
models = ['TimesNet', 'LightTS', 'DLinear', 'iTransformer', 'Informer', 'Transformer']
trainNums = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
data_paths = ["CropMapping_t30uxv", "CropMapping_t31tfj", "CropMapping_t31tfm", "CropMapping_t32ulu"]
periods = [1, 2, 3, 4]
command = [
  'python', '-u', '/home/aistudio/work/Time-Series-Library-main/run.py',
  '--task_name', 'classification',
  '--is_training', '1',
  '--root_path', '/home/aistudio/work/Time-Series-Library-main/dataset/'+data_path+'/',
  '--model_id', data_path+'_'+str(period)+'in4Period_'+str(trainNum)+'train',
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
  '--enc_in',  '3'
  ]
~~~

#### DCM
As an example, use the following command to run DCM method for all experients.
~~~
datasets = ['t30uxv', 't31tfj', 't31tfm', 't32ulu']
train_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
periods = [1, 2, 3, 4]
~~~


## Acknowledgement
The deep learning models implementations from this repository are based on the following github repositories.
- TempCNN, LSTM and InceptionTime from [BreizhCrops](https://github.com/dl4sits/BreizhCrops)
- TimesNet, LightTS, DLinear, iTransformer, Informer, Transformer from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- DCM from [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping)

