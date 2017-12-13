from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu', kernel_initializer='normal'))
model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=200, epochs=25)

print('\nTesting...')
test = model.evaluate(x_test, y_test)
print('Result: %.2f%%' % (test[1]*100))

save = input('\nSave this model? [y/n]: ')
if save == 'y':
    model.save('model.h5')
    print('Model saved')
