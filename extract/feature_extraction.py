from pathlib import Path
from tqdm import tqdm
import librosa
import logging
import numpy as np
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler


class Extractor:
    def __init__(self, config_file):
        """
        class used to create input data from audio file

        :param config_file: yaml file used for configuration
               must include:
                 - data_path: path to the directory where all datas are stored
                 - machines: list of all machine to be studied
                 - ae_dense: include the number of frame to concatenate and the number of mel to use
                 - ae-conv : include the number of frame to concatenate and the number of mel to use
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        self.data_path = Path(config_data["data_path"])
        if not self.data_path.is_dir():
            raise IOError(f"{self.data_path} is not a directory.")
        self.machines = config_data["machines"]
        self.ae_dense_param = config_data["ae_dense"]
        self.ae_conv_param = config_data["ae_conv"]

    ##############################################################################
    ##                             public method                                ##
    ##############################################################################

    def get_ae_dense_feature_train(self):
        """
        create the feature used for the dense autoencoder.
        For each machine store datas in a tfrecords file.

        If features have already been created, load them in a dataset and create batch of size 1024

        :return: name of the machine, train dataset (80% of the content), validation dataset (20% remaining)
        """
        for machine in self.machines:
            logging.info(machine)
            path_train = self.data_path / f"{machine}_train_dense.tfrecords"
            path_val = self.data_path / f"{machine}_val_dense.tfrecords"
            if not path_train.is_file():
                self.__create_ae_dense_feature_train(machine)

            raw_image_dataset = tf.data.TFRecordDataset(str(path_train))
            ds_train = raw_image_dataset.map(self.__parse_image_function)
            ds_train = ds_train.batch(512)
            ds_train.prefetch(1)

            raw_image_dataset = tf.data.TFRecordDataset(str(path_val))
            ds_val = raw_image_dataset.map(self.__parse_image_function)
            ds_val = ds_val.batch(512)
            ds_val.prefetch(1)

            yield machine, ds_train, ds_val

    def get_ae_conv_feature_train(self):
        """
        Create the feature used for the Convolutional Autoencoder training for each machine

        :return: name of the machine + numpy 4d array
        """
        for machine in self.machines:
            logging.info(machine)
            path_train = self.data_path / f"{machine}_train_conv.tfrecords"
            path_val = self.data_path / f"{machine}_val_conv.tfrecords"
            if not path_train.is_file():
                self.__create_ae_conv_feature_train(machine)

            raw_image_dataset = tf.data.TFRecordDataset(str(path_train))
            ds_train = raw_image_dataset.map(self.__parse_image_conv)
            ds_train = ds_train.batch(256)

            raw_image_dataset = tf.data.TFRecordDataset(str(path_val))
            ds_val = raw_image_dataset.map(self.__parse_image_conv)
            ds_val = ds_val.batch(256)

            yield machine, ds_train, ds_val

    def get_ae_dense_feature_test(self, machine, id_m):
        """
        create the input used for the dense autoencoder testing for each files and for specified
        machine

        :return: name of the machine + numpy 2d array
        """
        logging.info(machine)
        frames = self.ae_dense_param["frames"]
        n_mels = self.ae_dense_param["n_mels"]
        data_files = [fname for fname in self.get_files(machine, "test")
                      if str(fname.stem).split("_")[2] == id_m]

        for fname in tqdm(data_files):
            if "anomaly" in str(fname):
                label = 1
            else:
                label = 0
            yield label, self.__create_spectrogram(fname, frames, n_mels)

    def get_ae_conv_feature_test(self, machine):
        """
        Create the input used for the Convolutional Autoencoder testing for each files and for specified
        machine

        :return: name of the machine + numpy 4d array
        """
        frames = self.ae_conv_param["frames"]
        n_mels = self.ae_conv_param["n_mels"]
        data_files = self.get_files(machine, "test")

        for fname in data_files:
            if "anomaly" in str(fname):
                label = 1
            else:
                label = 0
            yield label, self.__create_spectrogram(fname, frames, n_mels)

    def get_files(self, machine, step):
        """
        extract all audio files stored for the machine and the step given

        :param machine: one of the machine stored
        :param step: must be train or test
        :return: all audio files stored
        """
        train_directory = Path(self.data_path / machine / step)

        return list(train_directory.glob("*.wav"))

    def get_id_machine(self, machine):
        """
        Each machine type has been tested on different specific machine.
        For a specific machine return this list, values are extracted from file name
        we assume name are : [anomaly|normal]_id_(number to be retuned)_.*

        :param machine: name of the machine
        :return:
        """
        data_files = self.get_files(machine, "test")

        return set([f.stem.split("_")[2] for f in data_files])

    ##############################################################################
    ##                             private method                               ##
    ##############################################################################

    def __create_ae_dense_feature_train(self, machine):
        """
        create the input used for the dense autoencoder training for each machine

        :return: name of the machine + numpy 2d array
        """
        frames = self.ae_dense_param["frames"]
        n_mels = self.ae_dense_param["n_mels"]
        path_train = self.data_path / f"{machine}_train_dense.tfrecords"
        path_val = self.data_path / f"{machine}_val_dense.tfrecords"
        logging.info("Extraction Data for training")
        data_files = self.get_files(machine, "train")
        data_files_train = data_files[:8*len(data_files)//10]
        data_files_val = data_files[8*len(data_files)//10:]

        with tf.io.TFRecordWriter(str(path_train)) as f:
            for fname in tqdm(data_files_train):
                for row in self.__create_spectrogram(fname, frames, n_mels):
                    record_bytes = self.__serialize_example(row, fname)
                    f.write(record_bytes)

        logging.info("Extraction Data for validation")
        with tf.io.TFRecordWriter(str(path_val)) as f:
            for fname in tqdm(data_files_val):
                for row in self.__create_spectrogram(fname, frames, n_mels):
                    record_bytes = self.__serialize_example(row, fname)
                    f.write(record_bytes)

    def __create_ae_conv_feature_train(self, machine):
        """
        create the input used for the convolutional autoencoder training for each machine

        :return: name of the machine + numpy 2d array
        """
        frames = self.ae_conv_param["frames"]
        n_mels = self.ae_conv_param["n_mels"]
        path_train = self.data_path / f"{machine}_train_conv.tfrecords"
        path_val = self.data_path / f"{machine}_val_conv.tfrecords"
        logging.info("Extraction Data for training")
        data_files = self.get_files(machine, "train")
        data_files_train = data_files[:8*len(data_files)//10]
        data_files_val = data_files[8*len(data_files)//10:]

        with tf.io.TFRecordWriter(str(path_train)) as f:
            for fname in tqdm(data_files_train):
                for row in self.__create_spectrogram(fname, frames, n_mels):
                    record_bytes = self.__serialize_example(row, fname)
                    f.write(record_bytes)

        logging.info("Extraction Data for validation")
        with tf.io.TFRecordWriter(str(path_val)) as f:
            for fname in tqdm(data_files_val):
                for row in self.__create_spectrogram(fname, frames, n_mels):
                    record_bytes = self.__serialize_example(row, fname)
                    f.write(record_bytes)

    def __create_spectrogram(self, fname, frames, n_mels):
        """
        create from each audio file given his melspectrogram.
        Each spectrogram is divided into 46 ms interval with a 50% overlap. 128 MFECs by frame are
        used with a windows of size 5

        :param fname: audio file to decompose
        :return: a 2d numpy array
        """
        sample, sr = librosa.load(fname)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=sample, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
        log_mel_spectrogram = 10. * np.log10(mel_spectrogram)

        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
        vector_array = np.zeros((vector_array_size, frames * n_mels))

        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        scaler = StandardScaler()
        vector_array = scaler.fit_transform(vector_array)

        return vector_array

    def __create_conv_spectrogram(self, fname):
        """
        Create from each audio file given its melspectrogram.
        Each spectrogram is divided into 64 ms interval with a 50% overlap.
        128 MFECs by frame are used .
        The melspectrogram is then segmented into 32 column data.

        :param fname: audio file to decompose
        :return: a 4d numpy array
        """
        frames = self.ae_conv_param["frames"]
        n_mels = self.ae_conv_param["n_mels"]

        vector_array = self.__create_spectrogram(fname, frames, n_mels)

        scaler = StandardScaler()
        X = scaler.fit_transform(vector_array)
        X = X.reshape((-1, frames, n_mels, 1))

        return X

    def __serialize_example(self, row, fname):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature = {
            'filename': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(fname).encode('utf-8')])),
            'row': tf.train.Feature(float_list=tf.train.FloatList(value=row))
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @tf.autograph.experimental.do_not_convert
    def __parse_image_function(self, example_proto):
        frames = self.ae_dense_param["frames"]
        n_mels = self.ae_dense_param["n_mels"]

        feature = {'row': tf.io.FixedLenFeature([frames*n_mels], tf.float32)}

        # Parse the input tf.train.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature)

        return parsed["row"], parsed["row"]

    @tf.autograph.experimental.do_not_convert
    def __parse_image_conv(self, example_proto):
        frames = self.ae_conv_param["frames"]
        n_mels = self.ae_conv_param["n_mels"]

        feature = {'row': tf.io.FixedLenFeature([frames * n_mels], tf.float32)}

        # Parse the input tf.train.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature)

        p = parsed["row"]
        p = tf.reshape(p, [frames, n_mels, 1])

        return p, p
