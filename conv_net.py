from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Convolution2D(32, size=(3, 3), activation='relu', input_shape=(28, 28, 3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
print(f"Accuracy: {scores[1]} %")
