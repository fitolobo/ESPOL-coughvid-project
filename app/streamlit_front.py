# Manejo de directorios y carga de datos
from pathlib import Path
import tempfile
# librerias para ejecutar el front-end
import streamlit as st
# libreria para manejo de dataframes
import pandas as pd

# AI
import torch
import torch.serialization
from utils import preproces_for_new_architecture
from models import CoughNetWithCNN

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Tos COVID-19",
    page_icon="🎤",
    layout="wide"
)


def main():
    st.title("Clasificador de Tos COVID-19 🦠")
    st.write("Sube un archivo de audio para analizar si la tos podría estar relacionada con COVID-19.")
    
    # Cargar el modelo
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            "checkpoints/best_model_cnn_20250128-200727.pth",
            map_location=device,
            weights_only=False  # Añadir esta línea
        )
        model = CoughNetWithCNN(len(checkpoint['hparams']['features']))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint, device
    
    try:
        model, checkpoint, device = load_model()
        hparams = checkpoint['hparams']
        scaler = checkpoint['scaler']
        st.success("Modelo cargado exitosamente! ✅")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return
    
    # Widget para subir archivo
    audio_file = st.file_uploader("Sube un archivo de audio", 
                                    type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
                                    help="Formatos soportados: WAV, MP3, OGG, M4A, WEBM")
            
    if audio_file is not None:
        # Crear un archivo temporal para guardar el audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        st.audio(audio_file, format='audio/wav')
        
        if st.button("Analizar Audio"):
            with st.spinner("Procesando audio..."):
                try:
                    # Extraer características
                    features = preproces_for_new_architecture(audio_path)
                    
                    # Preparar características escalares
                    scalar_features = []
                    for feature_name in hparams['features']:
                        if feature_name not in ['mel_spectrogram', 'chromagram']:
                            scalar_features.append(features[feature_name])
                    
                    # Convertir a DataFrame y escalar
                    df_features = pd.DataFrame(
                        [scalar_features], 
                        columns=[f for f in hparams['features'] 
                                if f not in ['mel_spectrogram', 'chromagram']]
                    )
                    scalar_x = torch.Tensor(scaler.transform(df_features))
                    
                    # Preparar espectrograma y cromagrama
                    spec_x = torch.FloatTensor(features['mel_spectrogram']).unsqueeze(0).unsqueeze(0)
                    chroma_x = torch.FloatTensor(features['chromagram']).unsqueeze(0).unsqueeze(0)
                    
                    # Realizar predicción
                    with torch.no_grad():
                        outputs = torch.softmax(model(scalar_x, spec_x, chroma_x), 1)
                        predictions = torch.argmax(outputs.data, 1)
                        
                        prob_sano = outputs[0][0].item()
                        prob_covid = outputs[0][1].item()
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Probabilidad de tener Covid-19",
                            value=f"{prob_covid*100:.2f}%"
                        )
                        # st.metric(
                        #     label="Probabilidad de COVID-19",
                        #     value=f"{prob_covid*100:.1f}%"
                        # )
                    
                    with col2:
                        if predictions.item() == 1:  # COVID
                            st.error("⚠️ Posible caso de COVID-19 detectado")
                        else:
                            st.success("✅ No se detectaron indicadores de COVID-19")
                    
                    # Mostrar formas de los tensores (para debugging)
                    if st.checkbox("Mostrar detalles técnicos"):
                        st.write(f"Scalar features shape: {scalar_x.shape}")
                        st.write(f"Spectrogram shape: {spec_x.shape}")
                        st.write(f"Chromagram shape: {chroma_x.shape}")
                    
                    st.info("""
                    **Nota**: Este es solo un sistema de ayuda y no debe ser usado como 
                    diagnóstico definitivo. Siempre consulte con un profesional de la salud.
                    """)
                    
                except Exception as e:
                    st.error(f"Error al procesar el audio: {str(e)}")
                
                # Limpiar archivo temporal
                Path(audio_path).unlink()

if __name__ == "__main__":
    main()