import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import du fichier classification personnalisé
from model.classification import (
    train_knn, train_nb, train_c45, train_rnn,
    evaluate_model, compare_models
)

# Import du fichier régression personnalisé
from model.regression import (
    train_simple_regression, train_multiple_regression,
    evaluate_regression, plot_simple_regression, 
    plot_multiple_regression, plot_regression_comparison
)


# Variables globales
train_dataset = None
trained_model = None
X_train = None
X_test = None
y_train = None
y_test = None
model_type = None
label_encoder_y = None
rnn_results = None
regression_type = None
current_algorithm = None
training_metrics = None


def prepare_data(df, target_column, test_size=0.2):
    """Préparer les données pour l'entraînement"""
    global X_train, X_test, y_train, y_test, model_type, label_encoder_y, regression_type

    if df is None or df.empty:
        return "❌ Dataset d'entraînement vide"

    if target_column not in df.columns:
        return f"❌ Colonne '{target_column}' inexistante"

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # Encoder les features catégorielles
    for col in X.select_dtypes(include='object'):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Déterminer le type de modèle
    if y.dtype == 'object' or y.nunique() < 20:
        model_type = 'classification'
        regression_type = None
        if y.dtype == 'object':
            label_encoder_y = LabelEncoder()
            y = label_encoder_y.fit_transform(y)
    else:
        model_type = 'regression'
        # Déterminer le type de régression
        if X.shape[1] == 1:
            regression_type = 'simple'
        else:
            regression_type = 'multiple'
        label_encoder_y = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    result_msg = (
        f"✅ Données prêtes\n"
        f"📌 Type : {model_type}"
    )
    
    if model_type == 'regression':
        result_msg += f" ({regression_type})"
    
    result_msg += (
        f"\n📌 Train : {len(X_train)} | Test : {len(X_test)}\n"
        f"📌 Features : {X_train.shape[1]}\n"
    )
    
    if model_type == 'classification':
        result_msg += f"📌 Classes : {y.nunique()}"
    else:
        result_msg += f"📌 Range y : [{y.min():.2f}, {y.max():.2f}]"

    return result_msg


def train_model_only(algorithm_name, k_value=5):
    """Entraîner le modèle SANS générer les visualisations"""
    global trained_model, X_train, X_test, y_train, y_test, model_type, rnn_results
    global regression_type, current_algorithm, training_metrics
    
    if X_train is None or y_train is None:
        return "❌ Veuillez d'abord préparer les données", None, None
    
    try:
        current_algorithm = algorithm_name
        
        # =============================
        # CLASSIFICATION
        # =============================
        if model_type == 'classification':
            if algorithm_name == 'KNN':
                trained_model = train_knn(X_train, y_train, k=k_value)
            elif algorithm_name == 'Naive Bayes':
                trained_model = train_nb(X_train, y_train)
            elif algorithm_name == 'C4.5':
                trained_model = train_c45(X_train, y_train)
            elif algorithm_name == 'Neural Network (MLP)':
                rnn_results, trained_model = train_rnn(X_train, y_train, X_test, y_test)
                results_df = pd.DataFrame(rnn_results)
                best_config = results_df.loc[results_df['Accuracy'].idxmax()]
                
                summary = (
                    f"✅ Réseau de Neurones entraîné\n"
                    f"=" * 50 + "\n\n"
                    f"🏆 MEILLEURE CONFIGURATION\n"
                    f"   Activation : {best_config['Activation']}\n"
                    f"   Neurones : {best_config['Nb_Neurones']}\n"
                    f"   Accuracy : {best_config['Accuracy']:.4f} ({best_config['Accuracy']*100:.2f}%)\n\n"
                    f"📊 Total configurations testées : {len(rnn_results)}\n\n"
                    f"💡 Cliquez sur 'Afficher les visualisations' pour voir les graphiques"
                )
                
                training_metrics = results_df
                return summary, results_df, None
            else:
                return f"❌ Algorithme '{algorithm_name}' non disponible", None, None
            
            # Prédictions
            y_pred_train = trained_model.predict(X_train)
            y_pred_test = trained_model.predict(X_test)
            
            # Métriques de classification
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            results = f"✅ Modèle entraîné : {algorithm_name}\n"
            results += "=" * 50 + "\n\n"
            results += f"📊 ACCURACY\n"
            results += f"   Train : {train_acc:.4f} ({train_acc*100:.2f}%)\n"
            results += f"   Test  : {test_acc:.4f} ({test_acc*100:.2f}%)\n\n"
            
            try:
                precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                
                results += f"📊 PRECISION : {precision:.4f}\n"
                results += f"📊 RECALL    : {recall:.4f}\n"
                results += f"📊 F1-SCORE  : {f1:.4f}\n"
            except:
                results += f"⚠️ Calcul des métriques détaillées non disponible\n"
            
            results += f"\n💡 Cliquez sur 'Afficher les visualisations' pour voir les graphiques"
            
            results_df = pd.DataFrame({
                'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Train': [f"{train_acc:.4f}", '-', '-', '-'],
                'Test': [f"{test_acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
            })
            
            training_metrics = results_df
        
        # =============================
        # RÉGRESSION
        # =============================
        else:
            if algorithm_name == 'Linear Regression (Simple)':
                if regression_type != 'simple':
                    return "❌ Régression simple nécessite une seule feature", None, None
                trained_model = train_simple_regression(X_train, y_train)
                
            elif algorithm_name == 'Linear Regression (Multiple)':
                if regression_type != 'multiple':
                    return "❌ Régression multiple nécessite plusieurs features", None, None
                trained_model = train_multiple_regression(X_train, y_train)
                
            else:
                return f"❌ Algorithme '{algorithm_name}' non disponible pour la régression", None, None
            
            # Évaluer le modèle
            metrics = evaluate_regression(trained_model, X_train, y_train, X_test, y_test)
            
            # Créer le résumé
            results = f"✅ Modèle entraîné : {algorithm_name}\n"
            results += "=" * 50 + "\n\n"
            
            # Afficher l'équation
            if hasattr(trained_model, 'get_equation'):
                results += f"📐 ÉQUATION:\n   {trained_model.get_equation()}\n\n"
            
            # Métriques
            results += f"📊 MSE (Mean Squared Error)\n"
            results += f"   Train : {metrics['train']['mse']:.4f}\n"
            results += f"   Test  : {metrics['test']['mse']:.4f}\n\n"
            
            results += f"📊 RMSE (Root Mean Squared Error)\n"
            results += f"   Train : {metrics['train']['rmse']:.4f}\n"
            results += f"   Test  : {metrics['test']['rmse']:.4f}\n\n"
            
            results += f"📊 R² SCORE\n"
            results += f"   Train : {metrics['train']['r2']:.4f}\n"
            results += f"   Test  : {metrics['test']['r2']:.4f}\n\n"
            
            results += f"📊 MAE (Mean Absolute Error)\n"
            results += f"   Train : {metrics['train']['mae']:.4f}\n"
            results += f"   Test  : {metrics['test']['mae']:.4f}\n\n"
            
            results += f"💡 Cliquez sur 'Afficher les visualisations' pour voir les graphiques"
            
            # Tableau des métriques
            results_df = pd.DataFrame({
                'Métrique': ['MSE', 'RMSE', 'R²', 'MAE'],
                'Train': [
                    f"{metrics['train']['mse']:.4f}",
                    f"{metrics['train']['rmse']:.4f}",
                    f"{metrics['train']['r2']:.4f}",
                    f"{metrics['train']['mae']:.4f}"
                ],
                'Test': [
                    f"{metrics['test']['mse']:.4f}",
                    f"{metrics['test']['rmse']:.4f}",
                    f"{metrics['test']['r2']:.4f}",
                    f"{metrics['test']['mae']:.4f}"
                ]
            })
            
            training_metrics = results_df
        
        return results, results_df, None
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erreur lors de l'entraînement : {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None


def show_visualizations():
    """Générer et afficher les visualisations du modèle entraîné"""
    global trained_model, X_train, X_test, y_train, y_test, model_type
    global rnn_results, regression_type, current_algorithm, training_metrics
    
    if trained_model is None:
        return None, "❌ Veuillez d'abord entraîner un modèle"
    
    try:
        # =============================
        # CLASSIFICATION
        # =============================
        if model_type == 'classification':
            from model.classification import create_classification_visualizations
            
            if current_algorithm == 'Neural Network (MLP)':
                plot_fig = create_classification_visualizations(
                    trained_model, X_train, y_train, X_test, y_test,
                    current_algorithm, results_df=None, rnn_results=rnn_results
                )
            else:
                plot_fig = create_classification_visualizations(
                    trained_model, X_train, y_train, X_test, y_test,
                    current_algorithm, results_df=training_metrics, rnn_results=None
                )
            
            return plot_fig, "✅ Visualisations générées avec succès"
        
        # =============================
        # RÉGRESSION
        # =============================
        else:
            if regression_type == 'simple':
                feature_name = X_train.columns[0] if isinstance(X_train, pd.DataFrame) else "X"
                plot_fig = plot_simple_regression(X_train, y_train, X_test, y_test, 
                                                  trained_model, feature_name)
            else:
                plot_fig = plot_multiple_regression(X_train, y_train, X_test, y_test, 
                                                    trained_model)
            
            return plot_fig, "✅ Visualisations générées avec succès"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erreur lors de la génération des visualisations : {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def compare_all_models():
    """Comparer tous les modèles disponibles avec visualisation"""
    global X_train, X_test, y_train, y_test, model_type
    
    if X_train is None or y_train is None:
        return "❌ Veuillez d'abord préparer les données", None, None
    
    if model_type != 'classification':
        return "❌ La comparaison n'est disponible que pour la classification", None, None
    
    try:
        models_to_compare = [
            'KNN', 'Naive Bayes', 'C4.5'
        ]
        
        results_df = compare_models(X_train, y_train, X_test, y_test, models_to_compare)
        
        summary = (
            f"✅ Comparaison de {len(results_df)} modèles terminée\n"
            f"=" * 50 + "\n\n"
            f"🏆 MEILLEUR MODÈLE\n"
            f"   {results_df.iloc[0]['Modèle']}\n"
            f"   Accuracy : {results_df.iloc[0]['Accuracy']:.4f}\n\n"
            f"📊 Résultats triés par Accuracy (décroissant)"
        )
        
        # Créer le graphique de comparaison
        from model.classification_viz import plot_model_comparison
        comparison_fig = plot_model_comparison(results_df)
        
        return summary, results_df, comparison_fig
        
    except Exception as e:
        return f"❌ Erreur lors de la comparaison : {str(e)}", None, None


def prediction_ui(dataset):
    """Interface utilisateur Gradio pour la prédiction"""
    
    with gr.Column():
        gr.Markdown("# 🤖 Modélisation & Prédiction Avancée")
        gr.Markdown("Interface intégrant classification et régression avec visualisations")

        # =============================
        # 📊 AFFICHAGE DATASETS
        # =============================
        with gr.Accordion("📂 Datasets chargés", open=False):
            train_df_view = gr.Dataframe(label="Dataset d'entraînement")
            dataset.change(lambda x: x, dataset, train_df_view)

        # =============================
        # 📊 PRÉPARATION DES DONNÉES
        # =============================
        with gr.Accordion("📊 Étape 1 : Préparation", open=True):
            target_column = gr.Dropdown(
                label="Colonne cible (Target)",
                choices=[]
            )

            dataset.change(
                fn=lambda df: gr.update(
                    choices=[] if df is None else list(df.columns)
                ),
                inputs=dataset,
                outputs=target_column
            )

            test_size_slider = gr.Slider(
                minimum=0.1,
                maximum=0.5,
                value=0.2,
                step=0.05,
                label="Taille du Test Set (%)"
            )

            prepare_btn = gr.Button("✅ Préparer les données", variant="primary")
            prepare_status = gr.Textbox(label="Statut", lines=6)

            prepare_btn.click(
                lambda df, col, size: prepare_data(df, col, size),
                inputs=[dataset, target_column, test_size_slider],
                outputs=prepare_status
            )

        # =============================
        # 🎯 ENTRAÎNEMENT
        # =============================
        with gr.Accordion("🎯 Étape 2 : Entraînement", open=True):
            algorithm_choice = gr.Dropdown(
                choices=[
                    'KNN', 
                    'Naive Bayes', 
                    'C4.5',
                    'Neural Network (MLP)',
                    'Linear Regression (Simple)',
                    'Linear Regression (Multiple)',
                ],
                value='KNN',
                label="Algorithme"
            )
            
            k_value = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="K pour KNN (si applicable)",
                visible=True
            )

            train_btn = gr.Button("🚀 Entraîner le modèle", variant="primary")

            results_text = gr.Textbox(label="Résultats", lines=15)
            results_table = gr.Dataframe(label="Tableau des métriques")
            rnn_details = gr.Dataframe(label="Détails RNN (si applicable)", visible=False)

            train_btn.click(
                train_model_only,
                inputs=[algorithm_choice, k_value],
                outputs=[results_text, results_table, rnn_details]
            )

        # =============================
        # 📈 VISUALISATIONS (NOUVEAU)
        # =============================
        with gr.Accordion("📈 Étape 3 : Visualisations", open=True):
            gr.Markdown("Générer les graphiques du modèle entraîné")
            
            show_viz_btn = gr.Button("📊 Afficher les visualisations", variant="secondary")
            viz_status = gr.Textbox(label="Statut", lines=2)
            plot_output = gr.Plot(label="Graphiques", visible=True)

            show_viz_btn.click(
                show_visualizations,
                inputs=[],
                outputs=[plot_output, viz_status]
            )

        # =============================
        # 📊 COMPARAISON DES MODÈLES
        # =============================
        with gr.Accordion("📊 Étape 4 : Comparaison des modèles", open=False):
            gr.Markdown("Comparer automatiquement tous les algorithmes de classification")
            
            compare_btn = gr.Button("🔍 Comparer tous les modèles", variant="secondary")
            compare_status = gr.Textbox(label="Résumé", lines=8)
            compare_table = gr.Dataframe(label="Résultats de comparaison")
            compare_plot = gr.Plot(label="Graphique de comparaison")

            compare_btn.click(
                compare_all_models,
                inputs=[],
                outputs=[compare_status, compare_table, compare_plot]
            )
