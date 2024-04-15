from comet_ml import Experiment
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import psutil
import GPUtil
import logging
class ConfusionMatrixLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions and true labels
        predictions = np.argmax(model.predict(valid_gen), axis=1)
        true_labels = np.array([])
        for _, labels in valid_gen:
            true_labels = np.concatenate([true_labels, np.argmax(labels, axis=1)])

        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predictions, normalize='true')

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Log the confusion matrix image to Comet.ml
        plt.savefig('confusion_matrix.png')
        experiment.log_image('confusion_matrix.png', name=f"Confusion_Matrix_Epoch_{epoch}")
        plt.close()
class ImageLoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Fetch a batch from the training data generator
        images, labels = next(iter(train_gen))
        # Log the first image in the batch to Comet.ml
        for i in range(min(10, len(images))):  # Log up to 10 images
            experiment.log_image(images[i], name=f"Epoch_{epoch}_Image_{i}")
            
class CometLogger(Callback):
    def __init__(self, experiment):
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            self.experiment.log_metric(metric, value, epoch=epoch)
            


# Initialize Comet.ml experiment
experiment = Experiment(
    api_key="SuxNYwm5fHHB03PDYwWZcG5eE",
    project_name="Shot Type Classification-1",
    workspace="m-majidi"
)

# Initialize Logging
logging.basicConfig(filename='training_log.log', level=logging.INFO)

# Set env variable for Comet.ml (if there's specific configuration needed)
os.environ['COMET_EVAL_LOG_CONFUSION_MATRIX'] = 'true'

# Custom data generator
class ImageLabelGenerator(Sequence):
    def __init__(self, image_dir, label_dir, image_size=(224, 224), batch_size=32, shuffle=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.image_filenames))
        self.label_map = {'[close-up]': 0, '[medium]': 1, '[full]': 2}
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_filenames = [self.image_filenames[k] for k in indexes]
        images, labels = self.__data_generation(batch_filenames)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_filenames):
        images = np.empty((self.batch_size, *self.image_size, 3))
        labels = np.empty((self.batch_size,), dtype=int)

        for i, filename in enumerate(batch_filenames):
            img_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')
            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image)
            image = preprocess_input(image)

            with open(label_path, 'r') as file:
                label = file.read().strip()  # Ensure it's correctly stripping any whitespace

            if label not in self.label_map:
                raise ValueError(f"Unrecognized label '{label}' in file '{label_path}'. Expected labels: {list(self.label_map.keys())}")
            
            images[i,] = image
            labels[i] = self.label_map[label]

        return images, tf.keras.utils.to_categorical(labels, num_classes=len(self.label_map))


# Load the ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freezing the layers
for layer in base_model.layers:
    layer.trainable = False

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Setup data generators
train_gen = ImageLabelGenerator('/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/train/images', '/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/train/labels')
valid_gen = ImageLabelGenerator('/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/valid/images', '/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/valid/labels')

test_gen = ImageLabelGenerator('/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/test/images', '/home/martin/Hockey/last-hockey-version/others/shottypes/DATASET-shot-types/test/labels', shuffle=False)



# Setup callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10)


# Custom callback to log GPU and CPU usage
class SystemMonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cpu_usage = psutil.cpu_percent()
        gpu_usage = GPUtil.getGPUs()[0].load * 100  # Convert fraction to percentage
        logs['cpu_usage'] = cpu_usage
        logs['gpu_usage'] = gpu_usage
        experiment.log_metrics(logs, epoch=epoch)
        logging.info(f"Epoch {epoch}: CPU usage: {cpu_usage}%, GPU usage: {gpu_usage}%")
        
# Train the model
# Train the model
model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stop,
        SystemMonitorCallback(),
        ImageLoggingCallback(),  # Log images from training batches
        ConfusionMatrixLoggingCallback()  # Log confusion matrix
    ]
)


# Evaluate the model on validation and test sets
val_results = model.evaluate(valid_gen)
test_results = model.evaluate(test_gen)

# Log validation and test results
experiment.log_metrics({"val_loss": val_results[0], "val_accuracy": val_results[1]})
experiment.log_metrics({"test_loss": test_results[0], "test_accuracy": test_results[1]})

# Save the fine-tuned model
model.save('fine_tuned_model.keras')

# End the experiment
experiment.end()
