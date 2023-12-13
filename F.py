import os
import sys
from PyQt5 import QtWidgets, QtGui
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from io import BytesIO

class AdvancedImageProcessor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        
    def init_ui(self):
        # Set up the GUI
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Advanced Image Processor')

        # Create GUI elements
        self.label = QtWidgets.QLabel('Enter Image URL:')
        self.url_input = QtWidgets.QLineEdit()
        self.process_button = QtWidgets.QPushButton('Process Image')
        self.result_label = QtWidgets.QLabel('Result:')

     
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.url_input)
        layout.addWidget(self.process_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # Connect button click event to the image processing function
        self.process_button.clicked.connect(self.process_image)

    def process_image(self):
        
        image_url = self.url_input.text()

        processed_image = self.load_and_process_image(image_url)

        # Display the processed image in the GUI
        self.display_image(processed_image)

    def load_and_process_image(self, image_url):
        # Load the image from URL using PIL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
       
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension


     
        img_np = img_tensor.numpy()

   
        img_tf = tf.convert_to_tensor(img_np)

      
        predictions = resnet_model.predict(img_tf)

    
        top_class = np.argmax(predictions)
        top_score = predictions[0, top_class]

        result_str = f'Top Prediction: Class {top_class}, Score: {top_score:.4f}'

        return result_str

    def display_image(self, result_str):
       
        self.result_label.setText(result_str)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AdvancedImageProcessor()
    window.show()
    sys.exit(app.exec_())
