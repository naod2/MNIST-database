import struct, random
from array import array
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.pipeline import Pipeline

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
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1
    plt.show()

# Show sample images
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)

# Reshape the data to 784 features
x_train = np.array(x_train) / 255
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

x_test = np.array(x_test) / 255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])


print("<<<END>>>")
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.1

# train a logistic regression model on the training set
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(x_train, y_train)
print (multi_model)
data_predictions = multi_model.predict(x_test)
print('Predicted labels: ', data_predictions[:15])
print('Actual labels   : ', y_test[:15])
from sklearn. metrics import classification_report

print(classification_report(y_test, data_predictions))
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Overall Accuracy:",accuracy_score(y_test, data_predictions))
print("Overall Precision:",precision_score(y_test, data_predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, data_predictions, average='macro'))
# Print the confusion matrix
from sklearn.metrics import confusion_matrix
mcm = confusion_matrix(y_test, data_predictions)
print(mcm)
import numpy as np
import matplotlib.pyplot as plt

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'], rotation=45)
plt.yticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'])
plt.xlabel("Predicted mnist")
plt.ylabel("Actual mnist")
plt.show()
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Get class probability scores



# Get ROC metrics for each class
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# fit the pipeline to train a linear regression model on the training set
multi_model = pipeline.fit(x_train, y_train)
print (multi_model)


# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, data_predictions[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

