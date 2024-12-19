
# compiling model using two hidden layers with relu activation and an output layer with softmax activation.
e = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
#ReLU is better due to its sparse activations
e.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history1 = e.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test))

# Plot accuracy vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), history1.history['accuracy'], 'bo-', label='Training Accuracy')
plt.title('Part (e): Accuracy vs Epochs ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
e.summary()

