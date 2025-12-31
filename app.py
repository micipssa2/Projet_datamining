import gradio as gr
import pandas as pd
from data_exploration import exploration_results
import signal
import sys

from prediction import prediction_ui
from data_preprocessing import preprocessing_ui
from model.cnn_model import cnn_ui
def gestionnaire_signal(sig, frame):
    print('\nFermeture propre du serveur...')
    sys.exit(0)

signal.signal(signal.SIGINT, gestionnaire_signal)

# =========================
# Fonction : charger dataset
# =========================
def load_dataset(file):
    if file is None:
        return None, "❌ Aucun fichier chargé"

    try:
        df = pd.read_csv(file)
        msg = f"✅ Dataset chargé avec succès\nLignes: {df.shape[0]} | Colonnes: {df.shape[1]}"
        return df, msg
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


# =========================
# Interface Gradio
# =========================
with gr.Blocks(title="Application Data Mining & ML") as demo:

    gr.Markdown(
        """
        # 📊 Application Machine Learning
        """
    )

    # État global (dataset partagé)
    dataset_state = gr.State(None)

    with gr.Tab("📂 Upload Dataset"):
        file_input = gr.File(label="Uploader un fichier CSV")
        load_btn = gr.Button("Charger le dataset")
        status_output = gr.Textbox(
            label="Statut",
            lines=5,       # nombre de lignes visibles (hauteur)
            max_lines=10,  # facultatif, limite si éditable
            interactive=False  # si tu veux juste afficher du texte
        )

        load_btn.click(
            fn=load_dataset,
            inputs=file_input,
            outputs=[dataset_state, status_output],
        )

    with gr.Tab("📈 Exploration"):
        explore_btn = gr.Button("🔍 Lancer l'exploration")

        head_df = gr.Dataframe(label="Aperçu")
        overview_df = gr.Dataframe(label="Dimensions & doublons")
        dtypes_df = gr.Dataframe(label="Types de données")
        missing_df = gr.Dataframe(label="Valeurs manquantes")
        unique_df = gr.Dataframe(label="Valeurs uniques")
        stats_df = gr.Dataframe(label="Statistiques descriptives")

        hist_plot = gr.Plot(label="Histogrammes")
        box_plot = gr.Plot(label="Boxplots")
        corr_plot = gr.Plot(label="Corrélation")

        explore_btn.click(
            fn=exploration_results,
            inputs=dataset_state,
            outputs=[
                head_df,
                overview_df,
                dtypes_df,
                missing_df,
                unique_df,
                stats_df,
                hist_plot,
                box_plot,
                corr_plot,
            ]
        )

    with gr.Tab("🧹 Prétraitement"):
        preprocessing_ui(dataset_state)

    with gr.Tab("🤖 Classification et regression"):
        prediction_ui(dataset_state)
    
    with gr.Tab("🧠 Classification CNN"):
        cnn_ui(dataset_state)

# Lancer l'application
if __name__ == "__main__":
    # Utilisez queue=False pour réduire les problèmes de threads
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)