import numpy as np
import struct
from array import array
from os import path, mkdir
from PIL import Image

# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

# Set file paths based on added MNIST Datasets
input_path = './input'
training_images_filepath = path.join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = path.join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = path.join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = path.join(input_path, 't10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Function of image creation
def save_images(index, images, label, folder_name):
    y = len(images)
    x = len(images[0])
    im = Image.new("L", (28, 28), 255)
    for j in range(y):
        for i in range(x):
            im.putpixel((i, j), int(images[j][i]))
    im.save("{}/{:1d}/{:05d}.png".format(folder_name, label, index))

# Create folders if not exist
dataset_type = ["train", "test"]
for entry in dataset_type:
    if not path.exists(entry):
        mkdir(entry)
    for i in range(0, 10):
        sub_folder_path = path.join(entry, str(i))
        if not path.exists(sub_folder_path):
            mkdir(sub_folder_path)

# Create image
for i in range(0, len(y_train)):
    save_images(i, x_train[i], y_train[i], "train")

for i in range(0, len(y_test)):
    save_images(i, x_test[i], y_test[i], "test")

print("<<<END>>>")
from tensorflow.keras import utils
x_train = np.array(x_train) / 255.0
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
print(x_train.shape)
y_train = np.array(y_train)

x_test = np.array(x_test) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
print(type(x_train), type(y_test))

# Define a CNN classifier network
img_size = (28,28)
batch_size = 20
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the model as a sequence of layers
model = Sequential()

# The input layer accepts an image and applies a convolution that uses 32 3x3 filters and a rectified linear unit activation function
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))

# Next we'll add a max pooling layer with a 2x2 patch
model.add(MaxPooling2D(pool_size=(2,2)))

# We can add as many layers as we think necessary - here we'll add another convolution and max pooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# And another set
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# A dropout layer randomly drops some nodes to reduce inter-dependencies (which can cause over-fitting)
model.add(Dropout(0.2))

# Flatten the feature maps 
model.add(Flatten())

# Generate a fully-connected output layer with a predicted probability for each class
# (softmax ensures all probabilities sum to 1)
model.add(Dense(10, activation='softmax'))

# With the layers defined, we can now compile the model for categorical (multi-class) classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
# Train the model over 5 epochs using 30-image batches and using the validation holdout dataset for validation
num_epochs = 10
history = model.fit(
    x_train, y_train,
    steps_per_epoch = 60000 // batch_size,
    validation_data = (x_test, y_test), 
    validation_steps = 10000// batch_size,
    epochs = num_epochs)
from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

classnames = ["0","1","2","3","4","5","6","7","8","9",]
import numpy as np
from sklearn.metrics import confusion_matrix


# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=85)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()