import os
import tempfile
import streamlit as st
import torch
from ultralytics import YOLO

# Configuração inicial
st.set_page_config(page_title="Classificador de Morfologia do Agregado", layout="wide")

# Carregar o modelo de classificação
device = torch.device('cpu')
model_path = 'models/best_yolov8_seg.pt'  # Substitua pelo caminho correto do seu modelo
model = YOLO(model_path)
model.to(device)

# Classes do modelo
class_names = ['arredondado', 'subalongado', 'alongado', 'bem alongado']

# Função para classificar imagens
def classify_image(image_path, model):
    results = model.predict(image_path)
    class_id = results[0].probs.top1  # ID da classe prevista
    confidence = results[0].probs.top1conf.item()  # Confiança da previsão
    return class_id, confidence

# Interface principal
def main():
    # Cabeçalho
    st.title("🌍 Classificador de Rochas - Grau de Esfericidade")
    st.subheader("🧪 Sistema para classificação morfológica de agregados.")
    st.markdown("---")

    # Barra lateral com instruções
    st.sidebar.title("📤 Selecione sua entrada")
    st.sidebar.info(
        "👋 **Instruções:**\n"
        "- Escolha o método de entrada.\n"
        "- Você pode capturar imagens com a câmera ou carregar arquivos existentes.\n"
        "- O resultado será exibido abaixo de cada imagem."
    )

    # Opção para usar a câmera ou carregar imagens
    use_camera = st.sidebar.checkbox("📸 Usar a câmera")
    uploaded_files = st.sidebar.file_uploader(
        "📂 Carregar imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    images_info = []  # Para armazenar os resultados

    if use_camera:
        st.markdown("### Tire uma foto usando sua câmera")
        camera_image = st.camera_input("Clique para capturar")
        if camera_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(camera_image.getvalue())
                image_path = temp_file.name

            # Classificar a imagem capturada
            class_id, confidence = classify_image(image_path, model)
            st.image(image_path, caption=f"Classe: {class_names[class_id]} ({confidence:.2%})", width=300)
            os.remove(image_path)  # Remover arquivo temporário

    elif uploaded_files:
        # Processar imagens carregadas
        st.info("⏳ Processando imagens carregadas...")
        columns = st.columns(3)  # Exibir em 3 colunas

        for idx, uploaded_file in enumerate(uploaded_files):
            with columns[idx % 3]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    image_path = temp_file.name

                # Classificar a imagem
                class_id, confidence = classify_image(image_path, model)
                st.image(image_path, caption=f"Classe: {class_names[class_id]} ({confidence:.2%})", width=200)
                os.remove(image_path)

    else:
        st.warning("⚠️ Nenhuma imagem carregada ou capturada. Por favor, escolha um método na barra lateral.")

    # Rodapé
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>"
        "Proposta de aplicação web para classificação morfológica de agregados usando Inteligência Artificial"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
