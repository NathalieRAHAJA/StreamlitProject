import numpy as np

from nn.classification_model import Classification
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


class Result:
    def __init__(self, machine, id_m, extractor):
        """
        class used to predict and calculate metriccs (f1 score, precision, recall) for the
        specific machine

        :param machine: name of the machine type
        :param id_m: instance of the machine
        :param extractor: class used to extract from the sound a spectogram
        :param threshold: number used to decide whether a sound is normal or not
        """
        self.machine = machine
        self.id_m = id_m
        self.threshold = extractor.threshold[machine][int(id_m)]

        self._model = Classification(len(extractor.output), "classification")
        self._extractor = extractor
        if machine == "valve":
            #self._model.load_weights(str(self._extractor.data_path)
                                    # + "/../DAS/save_model/no_log/classiBest")
            self._model.load_weights(str(self._extractor.data_path)
                                    + "C:/Users/Mimnat/Documents/DAS/save_model/no_log/classiBest")
            self.log = False
        else:
            #self._model.load_weights(str(self._extractor.data_path) +
                                     #"/../DAS/save_model/log/classiBest")
              self._model.load_weights(str(self._extractor.data_path)
                                    + "C:/Users/Mimnat/Documents/DAS/save_model/log/classiBest")
              self.log = True

        self.y_true = []
        self.y_pred = []

    def __repr__(self):
        """
        description of the class

        :return:
        """
        return f"result for classification model for {self.machine} {self.id_m}"

    @property
    def auc_score(self):
        """
        calculate auc score

        >>> from extract.feature_extraction_classification import ExtractorClassification
        >>> extractor = ExtractorClassification(r'../../etc/config.yaml')
        >>> result = Result("slider", "00", extractor)
        >>> result.y_true = [0, 0, 0, 1, 1, 1]
        >>> result.y_pred = [0.2, 0.98, 0.5, 0.99, 0.87, 0.2]
        >>> round(result.auc_score, 3)
        0.611

        :return: auc_score (float)
        """
        if not self.y_true:
            self.test()
        return roc_auc_score(self.y_true, self.y_pred)

    def f1_curve(self, iteration=100):
        """
        sweep threshold from 0 to 1 (by default 100 iterations) and calculate the f1 associated

        >>> from extract.feature_extraction_classification import ExtractorClassification
        >>> extractor = ExtractorClassification(r'../../etc/config.yaml')
        >>> result = Result("slider", "00", extractor)
        >>> result.y_true = [0, 0, 0, 1, 1, 1]
        >>> result.y_pred = [0.2, 0.98, 0.5, 0.99, 0.87, 0.2]
        >>> result.f1_curve(5)
        (array([0.  , 0.25, 0.5 , 0.75, 1.  ]), array([0.667, 0.571, 0.571, 0.667, 0.   ]))

        :return: x (list), y (np.array)
        """
        if not self.y_true:
            self.test()
        lx = np.linspace(0, 1, iteration)
        ly = []
        for x in lx:
            y_pred = [0 if p < x else 1 for p in self.y_pred]
            ly.append(round(f1_score(self.y_true, y_pred), 3))

        return lx, np.array(ly)

    def recall_curve(self, iteration=100):
        """
        sweep threshold from 0 to 1 (by default 100 iterations) and calculate the recall associated

        >>> from extract.feature_extraction_classification import ExtractorClassification
        >>> extractor = ExtractorClassification(r'../../etc/config.yaml')
        >>> result = Result("slider", "00", extractor)
        >>> result.y_true = [0, 0, 0, 1, 1, 1]
        >>> result.y_pred = [0.2, 0.98, 0.5, 0.99, 0.87, 0.2]
        >>> result.recall_curve(5)
        (array([0.  , 0.25, 0.5 , 0.75, 1.  ]), array([1.   , 0.667, 0.667, 0.667, 0.   ]))

        :return: x (list), y (np.array)
        """
        if not self.y_true:
            self.test()
        lx = np.linspace(0, 1, iteration)
        ly = []
        for x in lx:
            y_pred = [0 if p < x else 1 for p in self.y_pred]
            ly.append(round(recall_score(self.y_true, y_pred), 3))

        return lx, np.array(ly)

    def precision_curve(self, iteration=100):
        """
        sweep threshold from 0 to 1 (by default 100 iterations) and calculate the precision
        associated

        >>> from extract.feature_extraction_classification import ExtractorClassification
        >>> extractor = ExtractorClassification(r'../../etc/config.yaml')
        >>> result = Result("slider", "00", extractor)
        >>> result.y_true = [0, 0, 0, 1, 1, 1]
        >>> result.y_pred = [0.2, 0.98, 0.5, 0.99, 0.87, 0.2]
        >>> result.precision_curve(5)
        (array([0.  , 0.25, 0.5 , 0.75, 1.  ]), array([0.5  , 0.5  , 0.5  , 0.667, 0.   ]))

        :return: x (list), y (np.array)
        """
        if not self.y_true:
            self.test()
        lx = np.linspace(0, 1, iteration)
        ly = []
        for x in lx:
            y_pred = [0 if p < x else 1 for p in self.y_pred]
            ly.append(round(precision_score(self.y_true, y_pred, zero_division=0), 3))

        return lx, np.array(ly)

    def find_threshold(self, precision=0.9):
        """
        find the threshold which give a specific precision and the best f1 score

        >>> from extract.feature_extraction_classification import ExtractorClassification
        >>> extractor = ExtractorClassification(r'../../etc/config.yaml')
        >>> result = Result("slider", "00", extractor)
        >>> result.y_true = [0, 0, 0, 1, 1, 1]
        >>> result.y_pred = [0.2, 0.98, 0.5, 0.99, 0.87, 0.2]
        >>> result.find_threshold(0.6)
        >>> round(result.threshold, 3)
        0.505

        :return: threshold (float)
        """
        lx_accepted = [x for x, y in zip(*self.precision_curve()) if y > precision]
        lx, _ = zip(*sorted(zip(*self.f1_curve()), key=lambda l: l[1], reverse=True))
        for x in lx:
            if x in lx_accepted:
                self.threshold = x
                break
        else:
            self.find_threshold(precision-0.01)

    def test(self):
        """
        calculate for each sound stored in the test folder the prediction

        :return:
        """
        index_machine = self._extractor.output.index(f"{self.machine}_{int(self.id_m)}")
        for label, datas in self.__get_input():
            datas = datas.reshape(-1, 128, 128, 1)
            predict = np.mean(self._model.predict(datas), axis=0)[index_machine]

            self.y_true.append(label)
            self.y_pred.append(1 - predict)

    def predict(self, path):
        input_data = np.array(list(self._extractor.create_spectrogram(path, self.log)))
        datas = input_data.reshape(-1, 128, 128, 1)

        index_machine = self._extractor.output.index(f"{self.machine}_{int(self.id_m)}")
        predict = np.mean(self._model.predict(datas), axis=0)[index_machine]

        if 1 - predict < self.threshold:
            return 0
        else:
            return 1

    def __get_input(self):
        return self._extractor.get_ae_dense_feature_test(self.machine, self.id_m, self.log)
