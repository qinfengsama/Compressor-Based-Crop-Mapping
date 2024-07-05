# low-resource crop classification from multi-spectral time series using lossless compressors

## Dataset
Since the dataset is too large, you can download the data we use from the following links of google drive:
[Dataset](https://drive.google.com/drive/folders/1eMuwGf54EcDpi8Ed8mXVb0F9FnbxH5up?usp=sharing)

Once you have downloaded the data in `rs.npy` format and the ground truth in `gt.npy`, please place them in the `icpr/dataset_name/` path.

## Usage
### Dependencies
~~~
pip install pandas
pip install tqdm
pip install openpyxl
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

## Acknowledgement

All the experiment datasets are public, and we obtain them from the following links:
- [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

