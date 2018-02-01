from download_data import download_cifar_data
import pandas as pd

def download_data():
    return download_cifar_data()

def process_training_set(cifar_list):
    return stack_DataFrames(

        createDataFrame(cifar_list, 0),
        createDataFrame(cifar_list, 1),
        createDataFrame(cifar_list, 2),
        createDataFrame(cifar_list, 3),
        createDataFrame(cifar_list, 4),

        )

def process_test_set(cifar_list):
        return createDataFrame(cifar_list, 5)

def createDataFrame(cifar_list, batch_num):

    featureDataframe = pd.DataFrame.from_dict(
    cifar_list[batch_num][b'data'])

    labelDataframe = pd.DataFrame.from_dict(
    cifar_list[batch_num][b'labels'])

    return concat_DataFrame_axis_1(
        featureDataframe, labelDataframe
    )

def concat_DataFrame_axis_1(features, labels):
    return pd.concat([features, labels],
                    axis=1,
                    join_axes=[features.index])

def stack_DataFrames(d1, d2, d3, d4, d5):
    frames = [d1, d2, d3, d4, d5]
    return pd.concat(frames)

def training_set_to_nparray(data_frame):
    features = data_frame.iloc[:,:3072]
    labels   = data_frame.iloc[:,3072]

    features = features.as_matrix()
    labels   = labels.as_matrix()

    return features, labels

def test_set_to_nparry(data_frame):
    return training_set_to_nparray(data_frame)
