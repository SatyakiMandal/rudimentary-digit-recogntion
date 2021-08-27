import cv2 as cv #to import our own pibtures
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist=tf.keras.datasets.mnist #dictionary
(x_train,y_train), (x_test,y_test) = mnist.load_data() #splitting the dataset between training and testing

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_train, axis=1)
#normalize/scaling down the test and train data from easier computation
#NOTE: y terms are not scalled down because they are the lables

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')


img = cv.imread('2.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = np.expand_dims(img, 2)
img = np.invert(np.array([img]))

prediction = model.predict(img)
print(f'The result is probably: {np.argmax(prediction)}')

plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
