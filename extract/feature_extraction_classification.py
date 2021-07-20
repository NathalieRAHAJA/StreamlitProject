from pathlib import Path
from tqdm import tqdm
import librosa
import logging
import numpy as np
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler


class ExtractorClassification:
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
        #if not self.data_path.is_dir():
            #raise IOError(f"{self.data_path} is not a directory.")
        self.threshold = config_data["threshold"]
        self.machines = config_data["machines"].keys()
        self.id_machines = config_data["machines"]
        self.ae_dense_param = config_data["ae_dense"]
        self.output = []
        for k, v in config_data["machines"].items():
            self.output.extend([f"{k}_{id_m}" for id_m in v])

    ##############################################################################
    ##                             public method                                ##
    ##############################################################################

    def get_classi_feature_train(self):
        """
        create the feature used for the dense autoencoder.
        For each machine store datas in a tfrecords file.

        If features have already been created, load them in a dataset and create batch of size 1024

        :return: name of the machine,
                 train dataset (80% of the content),
                 validation dataset (20% remaining)
        """
        path_train = self.data_path / "train_classi_nolog.tfrecords"
        if not path_train.is_file():
            self.__create_classi_feature_train()

        # get number of file
        nb_files = self.__get_nb_of_files()
        train_size = int(0.8 * nb_files * 5)

        raw_image_dataset = tf.data.TFRecordDataset(str(path_train))
        ds = raw_image_dataset.map(self.__parse_image_function)
        ds = ds.shuffle(nb_files*5)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)

        train_ds = train_ds.batch(64)
        train_ds.prefetch(1)
        val_ds = val_ds.batch(64)
        val_ds.prefetch(1)

        return train_ds, val_ds

    def get_ae_dense_feature_test(self, machine, id_m, log):
        """
        create the input used for the classification testing for each files and for specified
        machine

        :return: name of the machine + numpy 2d array
        """
        logging.info(machine)
        data_files = [fname for fname in self.get_files(machine, "test")
                      if str(fname.stem).split("_")[2] == id_m]

        for fname in tqdm(data_files):
            if "anomaly" in str(fname):
                label = 1
            else:
                label = 0
            datas = list(self.create_spectrogram(fname, log))
            yield label, np.array(datas)

    def create_spectrogram(self, fname, log=True):
        """
        create from each audio file given his melspectrogram.
        Each spectrogram is divided into 46 ms interval with a 50% overlap. 128 MFECs by frame are
        used with a windows of size 5

        :param fname: audio file to decompose
        :return: a 2d numpy array
        """
        n_mels = self.ae_dense_param["n_mels"]
        sample, sr = librosa.load(fname)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=sample, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
        if log:
            mel_spectrogram = 10. * np.log10(mel_spectrogram)

        scaler = StandardScaler()
        vector_array = scaler.fit_transform(mel_spectrogram)
        for i in range(0, len(vector_array[0]) - 128, 4):
            arr = vector_array[:, i:128+i]

            yield arr.flatten()

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

    def __get_nb_of_files(self):
        """
        return the number of files which are used for this model

        :return:
        """
        return sum([len(self.get_files(machine, "train")) for machine in self.machines])

    def __create_classi_feature_train(self):
        """
        create the input used for the dense autoencoder training for each machine

        :return: name of the machine + numpy 2d array
        """
        path_train = self.data_path / "train_classi_nolog.tfrecords"
        logging.info("Extraction Data for training")
        with tf.io.TFRecordWriter(str(path_train)) as f:
            for machine in self.machines:
                logging.info(machine)
                data_files = self.get_files(machine, "train")

                for fname in tqdm(data_files):
                    for img in self.create_spectrogram(fname):
                        record_bytes = self.__serialize_example(img, fname, machine)
                        f.write(record_bytes)

    def __serialize_example(self, row, fname, machine):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        id_m = int(fname.stem.split("_")[2])
        output_name = f"{machine}_{id_m}"
        outputs_hot_encod = np.zeros(len(self.output), dtype=np.int8)
        outputs_hot_encod[self.output.index(output_name)] = 1

        feature = {
            'output': tf.train.Feature(int64_list=tf.train.Int64List(value=outputs_hot_encod)),
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[str(fname).encode('utf-8')])),
            'row': tf.train.Feature(float_list=tf.train.FloatList(value=row))
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @tf.autograph.experimental.do_not_convert
    def __parse_image_function(self, example_proto):
        n_mels = self.ae_dense_param["n_mels"]

        feature = {
            'output': tf.io.FixedLenFeature([len(self.output)], tf.int64),
            'row': tf.io.FixedLenFeature([16384], tf.float32)}

        # Parse the input tf.train.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature)

        parsed["row"] = tf.reshape(parsed["row"], [n_mels, n_mels, 1])

        return parsed["row"], parsed["output"]
