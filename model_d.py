# compling model

d = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(24, activation='sigmoid'),#input laye
    keras.layers.Dense(40, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')#output layer
])
#  model using two hidden layers with sigmoid activation and an output layer with softmax activation.
d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
history = d.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test))

# Ploting accuracy and epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.title('Part (d): Accuracy and Epochs 24, 40 Sigmoid activation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
d.summary()
