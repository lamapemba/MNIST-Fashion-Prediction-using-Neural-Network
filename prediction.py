import tensorflow as tf 
from tensorflow import keras   
import numpy as np   
import matplotlib.pyplot as plt  

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_name =  ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

###normalize the data into 0-1#####
train_images = tf.keras.utils.normalize(train_images, axis = 1)
test_images = tf.keras.utils.normalize(test_images, axis = 1)
print(train_images[7])

# plt.imshow(train_images[7], cmap = plt.cm.binary)
# plt.show()

###Create a neural network model ######

model = tf.keras.models.Sequential([
		keras.layers.Flatten(input_shape = (28, 28)),
		keras.layers.Dense(128, activation  =tf.nn.relu),
		keras.layers.Dense(128, activation = tf.nn.relu),
		keras.layers.Dense(10, activation = tf.nn.softmax)
	])

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs= 5)   #epochs: how many times the data should be iterate in the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)
for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap = plt.cm.binary)
	plt.xlabel("Actual: " + class_name[test_labels[i]])
	plt.title("Prediction: "+ class_name[np.argmax(prediction[i])])
	plt.show()


