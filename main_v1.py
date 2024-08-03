import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Carregar o modelo YOLOv8 a partir de um arquivo local
device = torch.device('cpu')
model_path = 'models/yolov8_model.pt'  # Substitua pelo caminho correto do seu modelo
model = YOLO(model_path)
model.to(device)

# Classes renomeadas
class_names = ['1-arredondado', '2-subalongado', '3-alongado', '4-bem_alongado']

# Definir um dicionário de cores para cada classe
class_colors = {
    '1-arredondado': (255, 0, 0),   # Red
    '2-subalongado': (0, 255, 0),   # Green
    '3-alongado': (0, 0, 255),      # Blue
    '4-bem_alongado': (255, 255, 0) # Yellow
}

# Função para segmentação de instâncias
def segment_image(image_path, model):
    # Realizar previsão
    results = model.predict(image_path)

    # Carregar a imagem original
    imagem_original = cv2.imread(image_path)
    class_counts = {name: 0 for name in class_names}
    class_confidences = {name: [] for name in class_names}
    segments_info = []

    # Processar cada máscara segmentada
    for i, mascara in enumerate(results[0].masks.xyn):
        # Obter as coordenadas x e y da máscara
        x = (mascara[:, 0] * imagem_original.shape[1]).astype("int")
        y = (mascara[:, 1] * imagem_original.shape[0]).astype("int")
        
        # Obter a classe detectada e incrementar a contagem
        class_id = int(results[0].boxes.cls[i].item())
        class_name = class_names[class_id]
        confidence = results[0].boxes.conf[i].item()
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)

        # Guardar informações do segmento para anotação
        segments_info.append({
            'coords': (x, y),
            'class_name': class_name,
            'confidence': confidence
        })

        # Definir a cor da máscara com base na classe
        cor_preenchimento = class_colors[class_name]

        # Desenhar a segmentação do YOLOv8 na imagem original
        imagem_transparente = np.zeros_like(imagem_original, dtype=np.uint8)
        cv2.polylines(imagem_transparente, [np.vstack((x, y)).T], isClosed=True, color=cor_preenchimento, thickness=2)
        cv2.fillPoly(imagem_transparente, [np.vstack((x, y)).T], color=cor_preenchimento)
        imagem_original = cv2.addWeighted(imagem_original, 1.0, imagem_transparente, 0.5, 0)

    # Salvar a imagem segmentada em um arquivo temporário
    output_file = os.path.join(os.getcwd(), "output_segmented.jpg")
    cv2.imwrite(output_file, imagem_original)

    return output_file, class_counts, class_confidences, segments_info

def main():
    st.title("Segmentação de Instâncias com YOLOv8")
    st.write("Carregue uma imagem para aplicar a segmentação de instâncias usando YOLOv8.")

    uploaded_files = st.file_uploader("Escolha as imagens...", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Salvar a imagem em um arquivo temporário
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                image_path = temp_file.name

            # Realizar a previsão e obter o resultado
            output_file, class_counts, class_confidences, segments_info = segment_image(image_path, model)

            # Verificar se o arquivo temporário existe
            if os.path.exists(output_file):
                image = open(output_file, "rb").read()
                st.image(image, caption=f"Imagem Segmentada - {uploaded_file.name}", use_column_width=True)
                
                # Exibir as classes detectadas com percentual de certeza
                st.write("Resultados de saída:")
                for class_name, count in class_counts.items():
                    if count > 0:
                        confidence_avg = np.mean(class_confidences[class_name]) * 100
                        st.write(f"{class_name.split('-')[1].capitalize()}: {confidence_avg:.2f}% de certeza")

                # Interface para revisão e ajuste manual
                st.write("Revise e ajuste as classificações manualmente:")
                for segment in segments_info:
                    coords = segment['coords']
                    original_class = segment['class_name']
                    confidence = segment['confidence']
                    st.write(f"Classe original: {original_class} com {confidence * 100:.2f}% de certeza")
                    new_class = st.selectbox("Selecione a nova classe:", class_names, index=class_names.index(original_class))
                    if st.button("Salvar alterações"):
                        segment['class_name'] = new_class
                        st.write(f"Classe alterada para {new_class}")

            else:
                st.write("Arquivo de saída não encontrado.")

            # Remover o arquivo temporário
            os.remove(image_path)
            os.remove(output_file)

if __name__ == "__main__":
    main()
