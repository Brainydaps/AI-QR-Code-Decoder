# Install the necessary libraries
!pip install autokeras tensorflow tf2onnx

import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import KFold
import tf2onnx
import onnx
import os

# Load the dataset
data_dir = '/kaggle/input/qrcodev1/qrcode/v1'
img_size = (290, 290)  # Adjust the image size if necessary
batch_size = 16  # Adjust batch size as needed

def decode_img(file_path):
    # Load and preprocess the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def get_label(file_path):
    # Extract the label (number) from the file name
    parts = tf.strings.split(file_path, os.sep)
    label_str = parts[-1]
    label_str = tf.strings.regex_replace(label_str, ".png", "")
    label = tf.strings.to_number(label_str, out_type=tf.float32)
    return label

def process_path(file_path):
    img = decode_img(file_path)
    label = get_label(file_path)
    return img, label

list_ds = tf.data.Dataset.list_files(str(data_dir + '/*.png'))
labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
labeled_ds = labeled_ds.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Extract features and labels
def split_features_labels(dataset):
    X, y = [], []
    for img, label in dataset:
        X.append(img)
        y.append(label)
    X = tf.concat(X, axis=0)
    y = tf.concat(y, axis=0)
    return X, y

X, y = split_features_labels(labeled_ds)

# Initialize the AutoKeras AutoModel for image regression
reg = ak.ImageRegressor(
    max_trials=10,  # Adjust max_trials as needed
    overwrite=True
)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_mae = []

for train_index, test_index in kf.split(X):
    # Convert indices to TensorFlow tensors
    X_train = tf.gather(X, train_index).numpy()
    X_test = tf.gather(X, test_index).numpy()
    y_train = tf.gather(y, train_index).numpy()
    y_test = tf.gather(y, test_index).numpy()
    
    # Train the model on the current fold
    reg.fit(X_train, y_train, epochs=20)
    
    # Evaluate the model on the current fold
    evaluation = reg.evaluate(X_test, y_test)
    
    # Extract the Mean Absolute Error (MAE)
    mae = evaluation[0]  # Assuming MAE is the first metric
    fold_mae.append(mae)
    print(f"Fold MAE: {mae}")

# Calculate and print the average MAE across all folds
average_mae = sum(fold_mae) / len(fold_mae)
print(f"Average Cross-Validation MAE: {average_mae}")

# Export the model to ONNX format
model = reg.export_model()
spec = (tf.TensorSpec([None, *img_size, 3], dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save the model
onnx.save(onnx_model, "autokeras_QRregressor.onnx")
