import os
import urllib.request
import pickle
file_name = "cifar-10.tgz"

def extract_tar(file_name=None):
    import tarfile
    with tarfile.open(name=file_name) as tarObj:
        tarObj.extractall()

def fileExists(file_name=None):
    return os.path.isfile(os.getcwd() + "/" + file_name )

def download_tgz_and_extract(string_url=None):

    if fileExists(file_name=file_name):
        print('file already exists')
    else:
        with urllib.request.urlopen(string_url) as response, open(file_name, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            extract_tar(file_name=file_name)
    return unpickle_batches()

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict

def unpickle_batches():
    data_batches = ["cifar-10-batches-py" + "/data_batch_1",
                    "cifar-10-batches-py" + "/data_batch_2",
                    "cifar-10-batches-py" + "/data_batch_3",
                    "cifar-10-batches-py" + "/data_batch_4",
                    "cifar-10-batches-py" + "/data_batch_5",
                    "cifar-10-batches-py" + "/test_batch"]
    dict_list = [unpickle(i) for i in data_batches]
    return dict_list

def download_cifar_data():
    string_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dict_list = download_tgz_and_extract(string_url=string_url)
    return dict_list
