import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QMessageBox,
                            QInputDialog, QProgressDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import cv2
import numpy as np
import pytesseract
from PIL import Image
import shutil

def get_tesseract_path():
    if getattr(sys, 'frozen', False):
        # Si estamos en un ejecutable
        base_path = sys._MEIPASS
        tesseract_path = os.path.join(base_path, 'tesseract', 'tesseract.exe')
    else:
        # Si estamos en desarrollo
        tesseract_path = 'tesseract'
    return tesseract_path

class NumberDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Números de Corredor")
        self.setGeometry(100, 100, 800, 600)
        
        # Configurar Tesseract
        pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Botones y elementos de la interfaz
        self.select_button = QPushButton("Seleccionar Imágenes")
        self.select_button.clicked.connect(self.select_images)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        
        # Añadir widgets al layout
        layout.addWidget(self.select_button)
        layout.addWidget(self.image_label)
        
        # Variables de estado
        self.current_image_path = None
        self.current_image = None
        self.image_list = []
        self.current_index = 0
        self.base_name = ""

    def select_images(self):
        # Solicitar el nombre base para las imágenes
        base_name, ok = QInputDialog.getText(
            self,
            'Nombre Base',
            'Ingrese el nombre base para las imágenes procesadas:'
        )
        
        if not ok or not base_name:
            return
            
        self.base_name = base_name
        
        # Seleccionar múltiples imágenes
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar Imágenes",
            "",
            "Imágenes (*.png *.jpg *.jpeg)"
        )
        
        if not file_names:
            return
            
        self.image_list = file_names
        self.current_index = 0
        
        # Crear directorio para imágenes etiquetadas si no existe
        output_dir = os.path.join(os.path.dirname(file_names[0]), 'etiquetadas')
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar las imágenes
        self.process_next_image()

    def process_next_image(self):
        if self.current_index >= len(self.image_list):
            # Mostrar resumen final
            QMessageBox.information(
                self,
                'Proceso Completado',
                f'Se han procesado {len(self.image_list)} imágenes.\n'
                f'Las imágenes etiquetadas se encuentran en la carpeta "etiquetadas".'
            )
            return
            
        self.current_image_path = self.image_list[self.current_index]
        self.current_image = cv2.imread(self.current_image_path)
        
        # Mostrar la imagen actual
        pixmap = QPixmap(self.current_image_path)
        scaled_pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        
        # Detectar número
        self.detect_number()

    def detect_number(self):
        if self.current_image is None:
            return
        
        try:
            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # Suavizado muy ligero
            blurred = cv2.GaussianBlur(gray, (3,3), 0)
            
            # Umbral que sabemos que funciona bien
            _, thresh = cv2.threshold(blurred, 115, 255, cv2.THRESH_BINARY)
            
            # Detectar bordes
            edges = cv2.Canny(thresh, 100, 200)
            
            # Dilatar para conectar dígitos
            kernel_dilate = np.ones((5,3), np.uint8)
            dilated = cv2.dilate(edges, kernel_dilate, iterations=2)
            
            # Cerrar gaps
            kernel_close = np.ones((3,5), np.uint8)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            results = []
            
            # Procesar regiones
            for contour in contours[:10]:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w > 30 and h > 20 and 0.5 <= w/h <= 4:
                    expand_x = int(w * 0.1)
                    expand_y = int(h * 0.1)
                    
                    start_x = max(0, x - expand_x)
                    start_y = max(0, y - expand_y)
                    end_x = min(gray.shape[1], x + w + expand_x)
                    end_y = min(gray.shape[0], y + h + expand_y)
                    
                    roi = blurred[start_y:end_y, start_x:end_x]
                    _, roi_thresh = cv2.threshold(roi, 115, 255, cv2.THRESH_BINARY)
                    
                    for config in [
                        '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789',
                        '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
                    ]:
                        text = pytesseract.image_to_string(roi_thresh, config=config)
                        numbers = ''.join(filter(str.isdigit, text))
                        
                        if numbers and 1 <= len(numbers) <= 3:
                            weight = 3 if len(numbers) == 2 else 1
                            results.append((numbers, weight))
            
            if results:
                # Contar frecuencias
                unique_numbers = {}
                for num, weight in results:
                    unique_numbers[num] = unique_numbers.get(num, 0) + weight
                
                # Ordenar por frecuencia
                sorted_numbers = sorted(unique_numbers.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)
                
                reply = QMessageBox.question(
                    self,
                    'Número Detectado',
                    f'Se detectó el número {sorted_numbers[0][0]}.\n'
                    f'¿Desea usar este número?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.save_image(sorted_numbers[0][0])
                else:
                    self.ask_manual_number()
            else:
                self.ask_manual_number()
                
        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'Error al procesar la imagen: {str(e)}'
            )
            self.ask_manual_number()

    def ask_manual_number(self):
        number, ok = QInputDialog.getText(
            self,
            'Entrada Manual',
            'No se detectó ningún número válido.\n'
            'Por favor, ingrese el número manualmente:'
        )
        
        if ok and number:
            if number.isdigit() and 1 <= len(number) <= 3:
                self.save_image(number)
            else:
                QMessageBox.warning(
                    self,
                    'Error',
                    'Por favor ingrese un número válido de 1-3 dígitos'
                )
                self.ask_manual_number()

    def save_image(self, number):
        try:
            # Crear nombre de archivo
            output_dir = os.path.join(os.path.dirname(self.current_image_path), 'etiquetadas')
            file_ext = os.path.splitext(self.current_image_path)[1]
            new_filename = f"{self.base_name}-{number}{file_ext}"
            new_path = os.path.join(output_dir, new_filename)
            
            # Copiar y renombrar la imagen
            shutil.copy2(self.current_image_path, new_path)
            
            # Procesar siguiente imagen
            self.current_index += 1
            self.process_next_image()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'Error al guardar la imagen: {str(e)}'
            )

def main():
    app = QApplication(sys.argv)
    window = NumberDetectorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 