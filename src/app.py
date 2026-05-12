import sys
import os
from pathlib import Path

# --- RESOLUCIÓN DE RUTAS PROFESIONAL ---
# Añadimos la raíz del proyecto al sys.path para que detecte el paquete 'src'
# Esto evita el ModuleNotFoundError: No module named 'src'
current_file = Path(__file__).resolve()
root_path = str(current_file.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.recommender import MovieRecommender

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Movie Recommender Pro", page_icon="🎬", layout="centered"
)


# --- LÓGICA DE NEGOCIO (CON CACHÉ) ---
@st.cache_resource
def load_and_train_system():
    """
    Carga los datos y entrena el modelo.
    Se ejecuta una sola vez y se guarda en memoria (cache).
    """
    try:
        loader = DataLoader()
        processor = TextPreprocessor()
        recommender = MovieRecommender()

        # 1. Carga
        df = loader.load_movies()
        if df is None or df.empty:
            return None, None

        # 2. Preprocesamiento
        processed_df = processor.create_soup(df)
        matrix = processor.get_feature_matrix(processed_df)

        # 3. Entrenamiento (Cálculo de Similitud Coseno)
        recommender.fit(processed_df, matrix)

        return df, recommender
    except Exception as e:
        # Esto aparecerá en tu terminal para debuguear
        print(f"DEBUG: Error en el pipeline: {e}")
        return None, None


# --- INTERFAZ DE USUARIO (UI) ---
def main():
    st.title("🎬 Movie Recommender System")
    st.markdown("""
    Este sistema analiza metadatos de películas (géneros y títulos) para encontrar 
    relaciones matemáticas en un espacio vectorial.
    """)

    # Intentar cargar el sistema
    movies_df, engine = load_and_train_system()

    # VALIDACIÓN CRÍTICA: Solo mostrar la UI si el motor cargó correctamente
    if movies_df is not None and engine is not None:
        st.success("✅ Motor de recomendación cargado y listo.")
        st.markdown("---")

        # Selector de películas
        movie_list = movies_df["title"].values
        selected_movie = st.selectbox(
            "Selecciona una película que te guste:",
            options=movie_list,
            index=None,
            placeholder="Escribe el nombre de una película...",
        )

        # Lógica de recomendación
        if selected_movie:
            if st.button("Encontrar películas similares"):
                with st.spinner("Calculando similitud coseno..."):
                    # Aquí ya es seguro llamar a engine.get_recommendations
                    recommendations = engine.get_recommendations(
                        selected_movie, top_n=5
                    )

                if recommendations:
                    st.subheader(f"Basado en '{selected_movie}', te recomendamos:")
                    for i, movie in enumerate(recommendations, 1):
                        st.success(f"**{i}.** {movie}")
                else:
                    st.warning(
                        "No pudimos encontrar similitudes suficientes para esta película."
                    )
    else:
        # Mensaje de error amigable si falla la carga
        st.error("❌ Error al inicializar el sistema.")
        st.info("""
            **Posibles soluciones:**
            1. Asegúrate de que los archivos `movies.csv` y `tags.csv` estén en `data/raw/`.
            2. Verifica que el archivo `src/config.py` tenga las rutas correctas.
            3. Revisa la terminal de VS Code para ver el error técnico detallado.
        """)

    # Sidebar informativo
    st.sidebar.title("Arquitectura del Proyecto")
    st.sidebar.info("""
    - **Algoritmo:** Content-Based Filtering.
    - **Métrica:** Cosine Similarity.
    - **NLP:** CountVectorizer (Bag of Words).
    - **Nivel:** Senior Portfolio Project.
    """)


if __name__ == "__main__":
    main()
