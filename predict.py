import numpy as np
from PIL import Image
import cv2 as cv
from tensorflow.keras.models import load_model


model = load_model('model.h5')

diseases = ["Melanoma","Nevus","Keratosis"]


p_image = Image.open('nerve.jpg')
image = p_image.resize((64, 64))  

image = np.array(image)
image = image / 255.0  
image = np.expand_dims(image, axis=0) 

predictions = model.predict(image)
predicted_class = np.argmax(predictions[0])
print(predicted_class)

print("Predicted Disease:", diseases[predicted_class])
p_image.show()