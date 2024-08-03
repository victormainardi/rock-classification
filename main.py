import os
import streamlit as st

# Importar outras bibliotecas
import cv2
import tempfile
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Definir a configuração da página para widescreen
st.set_page_config(layout="wide")

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
    best_confidence = 0
    best_segment = None
    best_class_name = None

    # Verificar se existem máscaras detectadas
    if results[0].masks is not None:
        # Processar cada máscara segmentada
        for i, mascara in enumerate(results[0].masks.xyn):
            # Obter a classe detectada e a confiança
            class_id = int(results[0].boxes.cls[i].item())
            class_name = class_names[class_id]
            confidence = results[0].boxes.conf[i].item()

            # Selecionar a previsão com a maior confiança
            if confidence > best_confidence:
                best_confidence = confidence
                best_segment = mascara
                best_class_name = class_name

        if best_segment is not None:
            # Obter as coordenadas x e y da melhor máscara
            x = (best_segment[:, 0] * imagem_original.shape[1]).astype("int")
            y = (best_segment[:, 1] * imagem_original.shape[0]).astype("int")

            # Definir a cor da máscara com base na classe
            cor_preenchimento = class_colors[best_class_name]

            # Desenhar a segmentação do YOLOv8 na imagem original
            imagem_transparente = np.zeros_like(imagem_original, dtype=np.uint8)
            cv2.polylines(imagem_transparente, [np.vstack((x, y)).T], isClosed=True, color=cor_preenchimento, thickness=2)
            cv2.fillPoly(imagem_transparente, [np.vstack((x, y)).T], color=cor_preenchimento)
            imagem_original = cv2.addWeighted(imagem_original, 1.0, imagem_transparente, 0.5, 0)

            # Adicionar texto de classificação e confiança
            label = f"{best_class_name.split('-')[1].capitalize()}: {best_confidence * 100:.2f}%"
            org = (x[0], y[0])  # Posição do texto
            cv2.putText(imagem_original, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_preenchimento, 2, cv2.LINE_AA)

    # Salvar a imagem segmentada em um arquivo temporário
    output_file = os.path.join(os.getcwd(), "output_segmented.jpg")
    cv2.imwrite(output_file, imagem_original)

    return output_file, best_class_name, best_confidence

def main():
    st.title("Classificador de Rochas - Grau de Esfericidade")
    st.subheader("Universidade Federal de Santa Maria")
    st.markdown("Carregue uma imagem para aplicar a segmentação de instâncias usando YOLOv8.")
    st.markdown("---")

    # Adicionar a opção de carregar arquivos
    st.markdown("### Escolha as imagens...")
    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
    
    # Adicionar a opção para usar a câmera
    use_camera = st.checkbox("Usar a câmera")

    # Mostrar o input da câmera apenas se o checkbox estiver marcado
    if use_camera:
        st.markdown("### Tire uma foto")
        camera_container = st.empty()
        camera_image = camera_container.camera_input("")

    if uploaded_files or (use_camera and camera_image):
        num_columns = 3  # Defina o número de colunas desejado
        columns = st.columns(num_columns)

        # Processar arquivos carregados
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                process_image(uploaded_file, i, columns, model)
        
        # Processar imagem da câmera
        if use_camera and camera_image:
            process_image(camera_image, len(uploaded_files) if uploaded_files else 0, columns, model)

    # Adicionar a observação no rodapé
    st.markdown("""
        <div style='position: fixed; bottom: 0; width: 100%; background-color: #262730; color: white; text-align: center; padding: 10px;'>
            Qualquer dúvida, entre em contato com a Equipe de Suporte Técnico do LAGEOLAM - Laboratório de Geologia Ambiental - <a href='mailto:haline.ceccato@gmail.com' style='color: #1f77b4;'>haline.ceccato@gmail.com</a>
        </div>
        """, unsafe_allow_html=True)

def process_image(image_file, index, columns, model):
    # Salvar a imagem em um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_file.getvalue())
        image_path = temp_file.name

    # Realizar a previsão e obter o resultado
    output_file, best_class_name, best_confidence = segment_image(image_path, model)

    # Verificar se o arquivo temporário existe
    if os.path.exists(output_file):
        image = open(output_file, "rb").read()
        
        # Mostrar a imagem segmentada na coluna correta
        col = columns[index % 3]
        with col:
            st.image(image, caption=f"Imagem Segmentada - {image_file.name if hasattr(image_file, 'name') else 'Câmera'}", use_column_width=True)
            
            # Exibir a melhor classe detectada com percentual de certeza
            st.markdown(f"**Saída:** {best_class_name.split('-')[1].capitalize()} - {best_confidence * 100:.2f}%")

            # Interface para revisão e ajuste manual
            st.markdown("Revise e ajuste as classificações manualmente")
            new_class = st.selectbox("Selecione a nova classe:", class_names, index=class_names.index(best_class_name), key=f"{image_file.name if hasattr(image_file, 'name') else 'camera'}-{index}")
            if st.button("Salvar alterações", key=f"save-{image_file.name if hasattr(image_file, 'name') else 'camera'}-{index}"):
                st.write(f"Classe alterada para {new_class}")
    else:
        st.write("Arquivo de saída não encontrado.")

    # Remover o arquivo temporário
    os.remove(image_path)
    os.remove(output_file)

if __name__ == "__main__":
    main()
