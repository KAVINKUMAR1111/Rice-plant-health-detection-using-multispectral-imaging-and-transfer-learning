import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen
import requests
from PIL import Image
from io import BytesIO
from PyQt5.QtCore import Qt

import numpy as np


class MobileWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDVI")
        self.setGeometry(100, 100, 360, 640)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.image_label = QLabel()
        self.image_label.setFixedSize(320, 320)
        self.layout.addWidget(self.image_label)
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_button)
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.process_image)
        self.layout.addWidget(self.predict_button)
        self.ndvi_label = QLabel("NDVI: ")
        self.layout.addWidget(self.ndvi_label)
        self.image_data = None

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image files (*.jpg *.png)")
        if file_path:
            with open(file_path, "rb") as f:
                self.image_data = f.read()
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaledToWidth(320)
            self.image_label.setPixmap(pixmap)
            f.close()  

    def process_image(self):
        if self.image_data:
            url = "http://127.0.0.1:5001/predict/"
            response = requests.post(url, files={"file": self.image_data})
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_array = np.array(img)
                red = img_array[:, :, 0]
                nir = img_array[:, :, 2]
                mask = (nir + red) == 0
                ndvi = np.where(mask, 0, (nir - red) / (nir + red))
                avg_ndvi = np.mean(ndvi)
                self.image_data = response.content

                pixmap = QPixmap()
                pixmap.loadFromData(self.image_data)
                pixmap = pixmap.scaledToWidth(320)
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.white, 2))
                width, height = pixmap.width(), pixmap.height()
                piece_width, piece_height = width // 6, height // 4

                for i in range(1, 6):
                    painter.drawLine(i * piece_width, 0, i * piece_width, height)
                for i in range(1, 4):
                    painter.drawLine(0, i * piece_height, width, i * piece_height)

                
                abnormal_pieces = []
                for row in range(4):
                    for col in range(6):
                        piece_ndvi = ndvi[row * piece_height: (row + 1) * piece_height,
                                        col * piece_width: (col + 1) * piece_width]
                        if np.mean(piece_ndvi) <= 0.3:
                            abnormal_pieces.append((row, col))

                for row, col in abnormal_pieces:
                    painter.setPen(QPen(Qt.red, 4))
                    x1, y1 = col * piece_width, row * piece_height
                    x2, y2 = (col + 1) * piece_width, (row + 1) * piece_height
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                painter.end()

                self.image_label.setPixmap(pixmap)
                if abnormal_pieces:
                    output_text = "Regions of Unhealthy plants (Mean NDVI < 0.5):\n"
                    for row, col in abnormal_pieces:
                        output_text += f"Image ({row}, {col})\n"
                    self.ndvi_label.setText(output_text)
                else:
                    self.ndvi_label.setText("No unhealthy region found")
            else:
                self.ndvi_label.setText("Error: Failed to fetch NDVI")
        else:
            self.ndvi_label.setText("Error: No image selected")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MobileWindow()
    window.show()
    sys.exit(app.exec_())