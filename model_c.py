# Defining the model 
c = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='sigmoid'),
    keras.layers.Dense(16, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])
#  model using two hidden layers with sigmoid activation and an output layer with softmax activation.
c.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
history = c.fit(x_train,
                y_train,
                epochs=8,
                validation_data=(x_test, y_test))

# Ploting accuracy and epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.title('Part (c) Accuracy vs Epochs two hidden layers ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
