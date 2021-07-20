import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from extract.feature_extraction_classification import ExtractorClassification
from model.result import Result
from nn.classification_model import Classification

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

extractor = ExtractorClassification(r'../etc/config.yaml')


def train_classificator():
    train_ds, val_ds = extractor.get_classi_feature_train()
    model = Classification(len(extractor.output), "classification")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=30,
            min_delta=0.001,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filepath=f"../save_model/no_log/classiBest",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )]
    model.fit(train_ds, epochs=100, callbacks=callbacks, validation_data=val_ds)


def test_classificator():
    lresult = OrderedDict()

    for machine in extractor.machines:
        for id_m in extractor.get_id_machine(machine):
            result = Result(machine, id_m, extractor)
            result.test()

            lresult.setdefault(machine, []).append(result)

    return lresult


if __name__ == "__main__":
    result = Result("slider", "02", extractor)
    out = result.predict(r"C:\Users\qfich\ML\projet_DS\data\slider\test\normal_id_02_00000009.wav")



