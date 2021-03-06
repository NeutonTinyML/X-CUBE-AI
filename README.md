![logo](repo_logo.png?raw=true "Title")
# Building Neuton TinyML models 

This research introduces [Neuton.ai service](https://neuton.ai) aimed at automatic creation of tinyML models for deployment on small microcontrollers.

Neuton AutoML framework model sizes had been benchmarked on 3 publicly available datasets. The task of each classification model was to identify the class to which the action belongs based on signal data.

For comparison purposes the models 6 traditional algorithms had been trained and validated on the same train/test splits, their accuracy and models sizes had been recorded.

Within each experiment models validation had been performed on out of sample test set.

## Datasets outline and preprocessing

### [Sussex-Huawei Locomotion (SHL)](http://www.shl-dataset.org)
A versatile annotated dataset of modes of locomotion and transportation of mobile users. It was recorded over a period of 7 months in 2017 by 3 participants engaging in 8 different modes of transportation in real-life setting in the United Kingdom. The goal is to classify one of 8 different activities: Still, Walk, Run, Bike, Car, Bus, Train, Subway.

Window with 64 sequential records had been selected to represent signal class.
Model training and validation had been performed on the subset of accelerometer, gyroscope, linear accelerometer features. The dataset had been ‘convolved’ with the defined window size extracting 9 different statistical values for each feature within a window: minimum, maximum, mean, rms, sign, var, PFD, skewness, kurtosis.

### [PAMAP2](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
Physical Activity Monitoring dataset contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor. The dataset can be used for activity recognition and intensity estimation, while developing and applying algorithms of data processing, segmentation, feature extraction and classification. The goal is to recognize which type of activity is taking place based on the sensor data.

Window with 512 sequential records had been selected to represent signal class.
The data contains a wide variety of different sensors data. For this experiment a subset of accelerometer data and heart rate sensor had been used.
Accelerometer features had been ‘convolved’ with the defined window size with extraction of statistical signal features based on the following research: See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/260291508
This has resulted in 108 features.

### [SisFall](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/)
A dataset of falls and activities of daily living (ADLs) acquired with a self-developed device composed of two types of accelerometer and one gyroscope. It consists of 19 ADLs and 15 fall types performed by 23 young adults, 15 ADL types performed by 14 healthy and independent participants over 62 years old, and data from one participant of 60 years old that performed all ADLs and falls. The goal is to train the model to recognize a binary target: is a person falling or not.

Window with 135 sequential records had been selected to represent signal class.
The dataset had been ‘convolved’ with the defined window size where the new training samples being comprised or all the values from the window in a single vector including the 9 statistical features of each window: minimum, maximum, mean, rms, sign, var, PFD, skewness, kurtosis.

## Models accuracy and size comparison

![comparison_table](comparison_table.png?raw=true "comparison_table")

###### *Measured by X-CUBE-AI
###### **ONNX model file size. Not measured by X-CUBE-AI
###### size is represented in Kilobytes


## Data & trained models
Preprocessed training and testing data along with the trained models for each use-case are available at the following links:
- [PAMAP2](https://1drv.ms/u/s!Al3jbxk-T6LAme9eA2-F3G_FhXN1Ew)
- [SHL](https://1drv.ms/u/s!Al3jbxk-T6LAme9fbnndAF6MyfRXIw)
- [SisFall](https://1drv.ms/u/s!Al3jbxk-T6LAme9gNShdx3UY-yxs8g)

## Models creation outline

### Neuton
Neuton models had been created via a free Automated TinyML platform [neuton.ai](https://neuton.ai).

### DecisionTree
Scikit-learn distribution of a DecisionTreeClassifier had been used with default settings.

```
from sklearn.tree import DecisionTeeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
pred = model.predict(test)
```

### AdaBoostClassifier
Scikit-learn distribution of a AdaBoostClassified had been used. n_estimators parameter had been slightly tweaked to achieve a higher accuracy. Different values of n_estimators had been trialed without a good impact on the accuracy. The smallest of the non-default n_estimators values had been selected.

```
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=200)
model.fit(X, y)
pred = model.predict(test)
```

### RandomForestClassifier
Scikit-learn distribution of a RandomForestClassifier had been used with default settings.

```
from sklearn.tree import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
pred = model.predict(test)
```

### KNNClassifier
Scikit-learn distribution of a KNeighborsClassifier had been used with default settings.

```
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, y)
pred = model.predict(test)
```

### GuassianNBClassifier
Scikit-learn distribution of a GaussianNB had been used with default settings.

```
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
pred = model.predict(test)
```

### Tensorflow Light
TF backend with Keras wrapper had been used for the purpose of this experiment. Two different architectures had been selected for the binary classfication use-case (SisFall) and for the multiclass classification use-cases (PAMAP2, SHL).

Optimal architectures in respect to the accuracy/size tradeof had been selected through various trials.

#### TF initialisation code
```
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
tf.__version__ # 2.4.1
keras.__version__ # 2.4.0

train = pd.read_csv('train.csv')
test_features = pd.read_csv('test.csv')
test_labels = pd.read_csv('labels.csv')

target = 'target_col_name'

train_features = train.drop(target,1)
train_labels = train[target]

scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
train_labels = np_utils.to_categorical(train_labels)

test_labels = encoder.transform(test_labels)
test_labels = np_utils.to_categorical(test_labels)

```

### binary tensorflow model architecture
```
def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='tanh'),
        layers.Dense(train[target].nunique(), activation = 'sigmoid')
        ])
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

tf.random.set_seed(8)

model = build_and_compile_model()
```

### multiclass tensorflow model architecture

```
def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(train[target].nunique(), activation = 'softmax')
        ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

tf.random.set_seed(5)

model = build_and_compile_model()
```

### Conversion of trained Tensorflow model to TensorflowLite
```
# load data
X_test = pd.read_csv('test.csv')
X_test = X_test.to_numpy()

# specify representative dataset
def representative_data_gen():
    for i_value in tf.data.Dataset.from_tensor_slices(X_test).batch(1).take(100):
        i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
        yield [i_value_f32]

#path to save your converted TF Lite model
converted_model_path_global = 'path_to_save_TF_model.tflite'

#path to saved TF model
converter = tf.lite.TFLiteConverter.from_saved_model('saved_TF_model_folder_path')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#Specify quantization type
converter.target_spec.supported_types = [tf.float32]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)

tflite_quant_model = converter.convert()
open(converted_model_path_global, 'wb').write(tflite_quant_model)
```

Please note that in this experiment for comparison purposes we are using the 32-bit TF model quantisation hence Neuton models had been deliberately built with 32-bit architecture.
