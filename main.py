import os
import tempfile
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(page_title="Classificador de Morfologia do Agregado", layout="wide")

device = torch.device('cpu')
model_path = 'models/best.pt'
model = YOLO(model_path)
model.to(device)

# Classes do modelo
class_names = ['arredondado', 'subalongado', 'alongado', 'bem alongado']

def classify_image(image_path, model):
    results = model.predict(image_path)
    boxes = results[0].boxes
    
    if boxes:
        highest_conf_index = torch.argmax(boxes.conf).item()
        class_id = int(boxes.cls[highest_conf_index].item())
        confidence = boxes.conf[highest_conf_index].item()
        return class_id, confidence
    else:
        return None, None

def main():
    st.title("🌍 Classificador de Rochas - Grau de Esfericidade")
    st.subheader("🧪 Sistema para classificação morfológica de agregados.")
    st.markdown("---")

    st.sidebar.title("📤 Selecione sua entrada")
    st.sidebar.info(
        "👋 **Instruções:**\n"
        "- Escolha o método de entrada.\n"
        "- Você pode capturar imagens com a câmera ou carregar arquivos existentes.\n"
        "- O resultado será exibido abaixo de cada imagem."
    )

    use_camera = st.sidebar.checkbox("📸 Usar a câmera")
    uploaded_files = st.sidebar.file_uploader(
        "📂 Carregar imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    images_info = []

    if use_camera:
        st.markdown("### Tire uma foto usando sua câmera")
        camera_image = st.camera_input("Clique para capturar")
        if camera_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(camera_image.getvalue())
                image_path = temp_file.name

            class_id, confidence = classify_image(image_path, model)
            if class_id is not None:
                st.image(image_path, caption=f"Classe: {class_names[class_id]} ({confidence:.2%})", width=300)
            else:
                st.image(image_path, caption="Nenhuma classe detectada", width=300)
            os.remove(image_path)

    elif uploaded_files:
        st.info("⏳ Processando imagens carregadas...")
        columns = st.columns(3)

        for idx, uploaded_file in enumerate(uploaded_files):
            with columns[idx % 3]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    image_path = temp_file.name

                class_id, confidence = classify_image(image_path, model)
                if class_id is not None:
                    st.image(image_path, caption=f"Classe: {class_names[class_id]} ({confidence:.2%})", width=200)
                else:
                    st.image(image_path, caption="Nenhuma classe detectada", width=200)
                os.remove(image_path)

    else:
        st.warning("⚠️ Nenhuma imagem carregada ou capturada. Por favor, escolha um método na barra lateral.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>"
        "Proposta de aplicação web para classificação morfológica de agregados usando Inteligência Artificial"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
