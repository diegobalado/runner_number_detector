import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Números de Corredor",
    page_icon="🏃",
    layout="wide"
)

# Título de la aplicación
st.title("Detector de Números de Corredor")

def detect_number(image):
    try:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
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
            
        return sorted_numbers[0][0]
        return None
            
    except Exception as e:
        st.error(f'Error al procesar la imagen: {str(e)}')
        return None

def save_image(image, number, base_name, file_ext, output_dir):
    try:
        new_filename = f"{base_name}-{number}{file_ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        # Convertir la imagen a formato PIL y guardarla
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(new_path)
        
        return new_filename
    except Exception as e:
        st.error(f'Error al guardar la imagen: {str(e)}')
        return None

def main():
    # Inicializar el estado de la sesión si no existe
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'base_name' not in st.session_state:
        st.session_state.base_name = ""
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    # Input para el nombre base
    base_name = st.text_input("Ingrese el nombre base para las imágenes procesadas:", 
                            value=st.session_state.base_name)
    if base_name:
        st.session_state.base_name = base_name

    # Uploader de imágenes
    uploaded_files = st.file_uploader("Seleccione las imágenes", 
                                    type=['png', 'jpg', 'jpeg'],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded_files and st.session_state.base_name:
        # Crear directorio para imágenes etiquetadas
        output_dir = "etiquetadas"
        os.makedirs(output_dir, exist_ok=True)

        # Mostrar progreso
        total_files = len(st.session_state.uploaded_files)
        processed_count = len(st.session_state.processed_files)
        st.progress(processed_count / total_files if total_files > 0 else 0)
        st.write(f"Procesando imagen {st.session_state.current_index + 1} de {total_files}")

        if st.session_state.current_index < len(st.session_state.uploaded_files):
            current_file = st.session_state.uploaded_files[st.session_state.current_index]
            
            if current_file.name not in st.session_state.processed_files:
                # Leer la imagen
                file_bytes = np.asarray(bytearray(current_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Mostrar la imagen
                st.image(image, caption=f"Imagen: {current_file.name}", use_column_width=True)
                
                # Detectar número
                detected_number = detect_number(image)
                
                if detected_number:
                    st.success(f"Número detectado: {detected_number}")
                    
                    # Crear dos columnas para los botones
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Confirmar número", key="confirm"):
                            new_filename = save_image(image, detected_number, 
                                                    st.session_state.base_name,
                                                    os.path.splitext(current_file.name)[1],
                                                    output_dir)
                            if new_filename:
                                st.info(f"Imagen guardada como: {new_filename}")
                                st.session_state.processed_files.add(current_file.name)
                                st.session_state.current_index += 1
                                st.rerun()
                    
                    with col2:
                        if st.button("Corregir número", key="correct"):
                            st.session_state.show_correction = True
                
                if st.session_state.get('show_correction', False):
                    manual_number = st.text_input(
                        "Ingrese el número correcto:",
                        key=f"manual_{current_file.name}"
                    )
                    
                    if manual_number and manual_number.isdigit() and 1 <= len(manual_number) <= 3:
                        new_filename = save_image(image, manual_number,
                                                st.session_state.base_name,
                                                os.path.splitext(current_file.name)[1],
                                                output_dir)
                        if new_filename:
                            st.info(f"Imagen guardada como: {new_filename}")
                            st.session_state.processed_files.add(current_file.name)
                            st.session_state.current_index += 1
                            st.session_state.show_correction = False
                            st.rerun()
                    elif manual_number:
                        st.warning("Por favor ingrese un número válido de 1-3 dígitos")
            else:
                st.session_state.current_index += 1
                st.rerun()

        if processed_count == total_files:
            st.success("¡Todas las imágenes han sido procesadas!")
            if st.button("Procesar nuevo lote"):
                st.session_state.current_index = 0
                st.session_state.processed_files = set()
                st.session_state.uploaded_files = []
                st.rerun()

if __name__ == "__main__":
    main() 