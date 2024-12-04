import os
import tempfile
import streamlit as st
import torch
from ultralytics import YOLO

# Configuração inicial
st.set_page_config(page_title="Classificador de Morfologia do Agregado", layout="wide")

# Carregar o modelo de classificação
device = torch.device('cpu')
model_path = 'C:/v12/runs/classify/train11/weights/best.pt'  # Atualize com o caminho do modelo de classificação
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
    st.subheader("🧪 Um sistema para auxiliar na classificação morfológica de amostras.")
    st.markdown("---")

    # Área de upload e instruções
    st.sidebar.title("📤 Carregue suas imagens")
    st.sidebar.info(
        "👋 **Instruções:**\n"
        "- Escolha suas imagens para análise.\n"
        "- Imagens suportadas: JPG, JPEG, PNG.\n"
        "- O resultado será exibido abaixo de cada imagem."
    )
    uploaded_files = st.sidebar.file_uploader(
        "Escolha suas imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    # Verificar se há imagens carregadas
    if uploaded_files:
        st.sidebar.success(f"🎉 {len(uploaded_files)} imagens carregadas.")
        
        # Mensagem de processamento
        st.info("⏳ Processando imagens...")

        # Criar colunas para organizar a exibição
        columns = st.columns(5)  # Divisão em 5 colunas para layout compacto

        for idx, uploaded_file in enumerate(uploaded_files):
            with columns[idx % 5]:
                # Salvar a imagem temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    image_path = temp_file.name

                # Classificar a imagem
                class_id, confidence = classify_image(image_path, model)

                # Renderizar a imagem (com largura reduzida)
                st.image(image_path, caption=f"Classe: {class_names[class_id]} ({confidence:.2%})", width=200)

                # Remover arquivo temporário
                os.remove(image_path)

    else:
        st.warning("⚠️ Nenhuma imagem carregada. Por favor, carregue imagens na barra lateral.")

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
