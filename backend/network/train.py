import numpy as np
import tensorflow as tf
from architecture import neuralNetwork

model = neuralNetwork()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(imgs, labels, epochs=10, batch_size=32)

score = model.evaluate(testImages, labels, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

