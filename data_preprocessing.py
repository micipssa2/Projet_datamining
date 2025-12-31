import pandas as pd
import numpy as np
import gradio as gr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
import cv2
from PIL import Image
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Variables globales pour l'historique
dataset_history = []
preprocessing_steps = []
current_step = 0


def get_columns(df):
    if df is None:
        return gr.Dropdown(choices=[])
    return gr.Dropdown(choices=list(df.columns))

def get_columns_num(df):
    if df is None:
        return gr.Dropdown(choices=[])
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return gr.Dropdown(choices=numeric_cols)




def delete_column(df, column_name):
    """Supprimer une colonne spécifique"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=[column_name])
    return df_copy, df_copy, f"✅ Colonne '{column_name}' supprimée"



def delete_rows_outliers_iqr(df, column_name):
    """Supprimer les valeurs aberrantes avec la méthode IQR"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df, df, "❌ La colonne doit être numérique"
    
    df_copy = df.copy()
    initial_rows = len(df_copy)
    
    Q1 = df_copy[column_name].quantile(0.25)
    Q3 = df_copy[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_copy = df_copy[(df_copy[column_name] >= lower_bound) & 
                      (df_copy[column_name] <= upper_bound)]
    
    deleted_rows = initial_rows - len(df_copy)
    return df_copy, df_copy, f"✅ {deleted_rows} valeurs aberrantes supprimées (IQR : [{lower_bound:.2f}, {upper_bound:.2f}])"

def delete_duplicates(df):
    """Supprimer tous les doublons"""
    if df is None:
        return df, df, "❌ Aucun jeu de données"
    df_copy = df.copy()
    initial_rows = len(df_copy)
    df_copy = df_copy.drop_duplicates()
    deleted = initial_rows - len(df_copy)
    return df_copy, df_copy, f"✅ {deleted} doublons supprimés"

def delete_missing_rows(df):
    """Supprimer toutes les lignes avec valeurs manquantes"""
    if df is None:
        return df, df, "❌ Aucun jeu de données"
    df_copy = df.copy()
    initial_rows = len(df_copy)
    df_copy = df_copy.dropna()
    deleted = initial_rows - len(df_copy)
    return df_copy, df_copy, f"✅ {deleted} lignes avec NaN supprimées"

def replace_value_custom(df, column_name, old_value, new_value):
    """Remplacer une valeur par une autre valeur personnalisée"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    df_copy = df.copy()
    try:
        count = (df_copy[column_name] == old_value).sum()
        df_copy[column_name] = df_copy[column_name].replace(old_value, new_value)
        return df_copy, df_copy, f"✅ {count} valeur(s) '{old_value}' remplacée(s) par '{new_value}'"
    except Exception as e:
        return df, df, f"❌ Erreur : {str(e)}"

def replace_value_with_mean(df, column_name, value_to_replace):
    """Remplacer une valeur par la moyenne de la colonne"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df, df, "❌ La colonne doit être numérique"
    
    df_copy = df.copy()
    
    try:
        if isinstance(value_to_replace, str):
            if value_to_replace.lower() == 'nan':
                value_to_replace = np.nan
            else:
                value_to_replace = float(value_to_replace)
        
        temp_series = df_copy[column_name].replace(value_to_replace, np.nan)
        mean_value = temp_series.mean(skipna=True)
        
        if pd.isna(mean_value):
            return df, df, "❌ Impossible de calculer la moyenne"
        
        df_copy[column_name] = df_copy[column_name].replace(value_to_replace, mean_value)
        
        count_replaced = (df[column_name] == value_to_replace).sum() if not pd.isna(value_to_replace) else df[column_name].isna().sum()
        
        return df_copy, df_copy, f"✅ {count_replaced} valeur(s) {value_to_replace} remplacée(s) par la moyenne : {mean_value:.4f}"
    except Exception as e:
        return df, df, f"❌ Erreur : {str(e)}"

def replace_value_with_median(df, column_name, value_to_replace):
    """Remplacer une valeur par la médiane"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df, df, "❌ La colonne doit être numérique"
    
    df_copy = df.copy()
    try:
        median_value = df_copy[column_name].median()
        count = (df_copy[column_name] == value_to_replace).sum()
        df_copy[column_name] = df_copy[column_name].replace(value_to_replace, median_value)
        return df_copy, df_copy, f"✅ {count} valeur(s) remplacée(s) par la médiane : {median_value:.4f}"
    except Exception as e:
        return df, df, f"❌ Erreur : {str(e)}"

def replace_outliers_with_iqr(df, column_name):
    """Remplacer les valeurs aberrantes par les bornes IQR"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df, df, "❌ La colonne doit être numérique"
    
    df_copy = df.copy()
    
    Q1 = df_copy[column_name].quantile(0.25)
    Q3 = df_copy[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df_copy[column_name] < lower_bound) | (df_copy[column_name] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    df_copy.loc[df_copy[column_name] < lower_bound, column_name] = lower_bound
    df_copy.loc[df_copy[column_name] > upper_bound, column_name] = upper_bound
    
    return df_copy, df_copy, f"✅ {outliers_count} valeurs aberrantes remplacées par les bornes [{lower_bound:.2f}, {upper_bound:.2f}]"

def replace_value_with_mode(df, column_name, value_to_replace):
    """Remplacer une valeur par le mode (pour variables catégorielles)"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    df_copy = df.copy()
    
    try:
        mode_value = df_copy[column_name].mode()
        if len(mode_value) == 0:
            return df, df, "❌ Impossible de calculer le mode"
        
        mode_value = mode_value[0]
        count_before = (df_copy[column_name] == value_to_replace).sum()
        df_copy[column_name] = df_copy[column_name].replace(value_to_replace, mode_value)
        
        return df_copy, df_copy, f"✅ {count_before} valeur(s) '{value_to_replace}' remplacée(s) par le mode : '{mode_value}'"
    except Exception as e:
        return df, df, f"❌ Erreur : {str(e)}"

def fill_missing_with_strategy(df, column_name, strategy='mean'):
    """Remplir les valeurs manquantes avec différentes stratégies"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    df_copy = df.copy()
    initial_nulls = df_copy[column_name].isnull().sum()
    
    if pd.api.types.is_numeric_dtype(df_copy[column_name]):
        if strategy == 'mean':
            fill_value = df_copy[column_name].mean()
        elif strategy == 'median':
            fill_value = df_copy[column_name].median()
        elif strategy == 'mode':
            fill_value = df_copy[column_name].mode()[0] if len(df_copy[column_name].mode()) > 0 else 0
        elif strategy == 'zero':
            fill_value = 0
    else:
        if strategy == 'mode':
            fill_value = df_copy[column_name].mode()[0] if len(df_copy[column_name].mode()) > 0 else 'Inconnu'
        else:
            fill_value = 'Inconnu'
    
    df_copy[column_name].fillna(fill_value, inplace=True)
    return df_copy, df_copy, f"✅ {initial_nulls} NaN remplis par {strategy} : {fill_value}"


def normalize_columns(df, columns_str=None, method='standard'):
    """Normaliser des colonnes spécifiques avec différentes méthodes"""
    if df is None:
        return df, df, "❌ Aucun jeu de données"
    
    df_copy = df.copy()
    
    if columns_str:
        columns = [col.strip() for col in columns_str.split(',')]
        columns = [col for col in columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]
    else:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return df, df, "❌ Aucune colonne numérique valide"
    
    for col in columns:
        if method == 'standard':
            scaler = StandardScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
        elif method == 'robust':
            scaler = RobustScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
        elif method == 'log':
            min_val = df_copy[col].min()
            if min_val <= 0:
                df_copy[col] = np.log(df_copy[col] - min_val + 1)
            else:
                df_copy[col] = np.log(df_copy[col])
        elif method == 'zscore_manual':
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            if std > 0:
                df_copy[col] = (df_copy[col] - mean) / std
    
    return df_copy, df_copy, f"✅ {len(columns)} colonne(s) normalisée(s) avec la méthode : {method}"

def encode_multiple_columns(df, columns_str, method='label'):
    """Encoder plusieurs colonnes catégorielles"""
    if df is None:
        return df, df, "❌ Aucun jeu de données"
    
    df_copy = df.copy()
    columns = [col.strip() for col in columns_str.split(',')]
    encoded_cols = []
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        if method == 'label':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoded_cols.append(col)
        
        elif method == 'onehot':
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy = df_copy.drop(columns=[col])
            encoded_cols.append(f"{col}_onehot")
        
        elif method == 'binary':
            unique_vals = df_copy[col].unique()
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df_copy[col] = df_copy[col].map(mapping)
                encoded_cols.append(col)
    
    if not encoded_cols:
        return df, df, "❌ Aucune colonne valide pour l'encodage"
    
    return df_copy, df_copy, f"✅ {len(encoded_cols)} colonne(s) encodée(s) : {', '.join(encoded_cols)}"

def create_bins(df, column_name, n_bins=5, strategy='equal'):
    """Créer des intervalles pour une variable numérique"""
    if df is None or column_name not in df.columns:
        return df, df, "❌ Colonne introuvable"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df, df, "❌ La colonne doit être numérique"
    
    df_copy = df.copy()
    new_col_name = f"{column_name}_intervalles"
    
    try:
        if strategy == 'equal':
            df_copy[new_col_name] = pd.cut(df_copy[column_name], bins=n_bins, labels=False)
        elif strategy == 'quantile':
            df_copy[new_col_name] = pd.qcut(df_copy[column_name], q=n_bins, labels=False)
        elif strategy == 'custom':
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(df_copy[column_name].dropna(), percentiles)
            df_copy[new_col_name] = pd.cut(df_copy[column_name], bins=bins, labels=False)
        
        return df_copy, df_copy, f"✅ Création d'intervalles appliquée : {n_bins} intervalles avec stratégie '{strategy}'"
    except Exception as e:
        return df, df, f"❌ Erreur de création d'intervalles : {str(e)}"




def preprocess_image_advanced(image_path, operations):
    """Prétraitement avancé d'image avec opérations multiples"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Image introuvable"
        
        operations_applied = []
        ops_list = [op.strip() for op in operations.split(',')]
        
        for op in ops_list:
            if op == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                operations_applied.append('Niveaux de gris')
            elif op == 'resize_224':
                img = cv2.resize(img, (224, 224))
                operations_applied.append('Redimensionnement 224x224')
            elif op == 'resize_128':
                img = cv2.resize(img, (128, 128))
                operations_applied.append('Redimensionnement 128x128')
            elif op == 'normalize':
                img = img.astype('float32') / 255.0
                operations_applied.append('Normalisation [0,1]')
            elif op == 'standardize':
                if len(img.shape) == 3:
                    for i in range(3):
                        img[:,:,i] = (img[:,:,i] - img[:,:,i].mean()) / img[:,:,i].std()
                else:
                    img = (img - img.mean()) / img.std()
                operations_applied.append('Standardisation')
            elif op == 'edge_detection':
                img = cv2.Canny(img, 100, 200)
                operations_applied.append('Détection de contours')
            elif op == 'blur':
                img = cv2.GaussianBlur(img, (5, 5), 0)
                operations_applied.append('Flou gaussien')
            elif op == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
                operations_applied.append('Accentuation')
        
        return img, f"✅ Opérations appliquées : {', '.join(operations_applied)}"
    except Exception as e:
        return None, f"❌ Erreur : {str(e)}"

def batch_process_images(input_folder, output_folder, operations, file_format='jpg'):
    """Traiter un dossier d'images en lot"""
    if not os.path.exists(input_folder):
        return "❌ Dossier source introuvable"
    
    os.makedirs(output_folder, exist_ok=True)
    processed = 0
    failed = 0
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            input_path = os.path.join(input_folder, filename)
            
            try:
                img, _ = preprocess_image_advanced(input_path, operations)
                
                if img is not None:
                    name, ext = os.path.splitext(filename)
                    output_filename = f"{name}_traite.{file_format}"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, img)
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
    
    return f"✅ Traitement en lot terminé : {processed} images traitées, {failed} échecs"



def undo_last_operation(df):
    """Annuler la dernière opération"""
    global dataset_history, current_step
    if len(dataset_history) > 1:
        dataset_history.pop()
        current_step -= 1
        return dataset_history[-1] if dataset_history else df, f"↩️ Opération annulée. {len(dataset_history)} étapes restantes"
    return df, "❌ Plus d'opérations à annuler"



# =============================
# 🎨 INTERFACE GRADIO
# =============================

def preprocessing_ui(dataset_state):
    with gr.Column():
        gr.Markdown("# 🧹 Prétraitement Manuel des Données - Version Avancée")
        gr.Markdown("### Contrôle total sur chaque étape de prétraitement")
        
        # =============================
        # 📊 CONTRÔLE D'HISTORIQUE
        # =============================
        with gr.Accordion("⏮️ Contrôle d'Historique", open=False):
            with gr.Row():
                undo_btn = gr.Button("↩️ Annuler dernière opération")
                redo_btn = gr.Button("↪️ Rétablir")
                clear_history_btn = gr.Button("🗑️ Vider l'historique")
            history_display = gr.Textbox(label="Historique des opérations", lines=3, interactive=False)
            undo_btn.click(undo_last_operation, inputs=dataset_state, outputs=[dataset_state, history_display])
        
        # =============================
        # 🗑️ SUPPRESSION AVANCÉE
        # =============================
        with gr.Accordion("🗑️ Suppression Avancée", open=True):
            gr.Markdown("### Supprimer des colonnes")
            with gr.Row():
                col_to_delete = gr.Dropdown(
                label="Colonne à supprimer",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs=col_to_delete
                )
                delete_col_btn = gr.Button("❌ Supprimer cette colonne")
            
            
            
            gr.Markdown("### Gestion des valeurs aberrantes")
            with gr.Row():
                col_outliers =gr.Dropdown(
                label="Colonne numérique",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns_num,
                    inputs=dataset_state,
                    outputs=col_outliers
                ) 
                
                delete_outliers_btn = gr.Button("📊 Supprimer valeurs aberrantes (méthode IQR)")
            
            with gr.Row():
                delete_dup_btn = gr.Button("🧹 Supprimer tous les doublons")
                delete_nan_btn = gr.Button("❌ Supprimer toutes les lignes avec NaN")
            
            delete_output = gr.Dataframe(label="Jeu de données après suppression")
            status_delete = gr.Textbox(label="Statut")
        
        # =============================
        # 🔄 REMPLACEMENT AVANCÉ
        # =============================
        with gr.Accordion("🔄 Remplacement de Valeurs Avancé", open=False):
            gr.Markdown("### Remplacer des valeurs spécifiques")
            with gr.Row():
                col_replace = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs=col_replace
                )
                old_val = gr.Textbox(label="Valeur à remplacer", placeholder="Ex : 0 ou 'unknown'")
                new_val = gr.Textbox(label="Nouvelle valeur", placeholder="Laissez vide pour méthodes automatiques")
                replace_custom_btn = gr.Button("✏️ Remplacer par valeur personnalisée")
            
            gr.Markdown("### Méthodes automatiques")
            with gr.Row():
                col_auto = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs=col_auto
                )
                value_to_replace = gr.Textbox(label="Valeur à remplacer", placeholder="Ex : -1")
                with gr.Column():
                    replace_mean_btn = gr.Button("📊 Par moyenne")
                    replace_median_btn = gr.Button("📊 Par médiane")
                    replace_mode_btn = gr.Button("📊 Par mode")
                    replace_outliers_iqr_btn = gr.Button("📈 Remplacer valeurs aberrantes (IQR)")
            
            replace_output = gr.Dataframe(label="Jeu de données après remplacement")
            status_replace = gr.Textbox(label="Statut")
        
        # =============================
        # 🧹 GESTION DES NaN
        # =============================
        with gr.Accordion("🧹 Gestion Avancée des Valeurs Manquantes", open=False):
            gr.Markdown("### Remplir les NaN par colonne")
            with gr.Row():
                col_fill = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs=col_fill
                )
                with gr.Column():
                    fill_mean_btn = gr.Button("📊 Moyenne")
                    fill_median_btn = gr.Button("📊 Médiane")
                    fill_mode_btn = gr.Button("📊 Mode")
                    fill_zero_btn = gr.Button("0️⃣ Zéro")
            
            fill_output = gr.Dataframe(label="Jeu de données après remplissage")
            status_fill = gr.Textbox(label="Statut")
        
        # =============================
        # ⚡ NORMALISATION
        # =============================
        with gr.Accordion("⚡ Normalisation & Transformation", open=False):
            gr.Markdown("### Méthodes de normalisation")
            with gr.Row():
                norm_cols = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs = norm_cols
                )
                with gr.Column():
                    norm_standard_btn = gr.Button("📊 Standard (Z-score)")
                    norm_minmax_btn = gr.Button("📊 Min-Max (0-1)")
                    norm_robust_btn = gr.Button("🛡️ Robuste (résistant aux aberrations)")
                    norm_log_btn = gr.Button("📈 Transformation logarithmique")
            
            gr.Markdown("### Discrétisation en intervalles")
            with gr.Row():
                bin_col = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns_num,
                    inputs=dataset_state,
                    outputs=bin_col
                )
                n_bins = gr.Slider(minimum=2, maximum=20, value=5, label="Nombre d'intervalles")
                with gr.Column():
                    bin_equal_btn = gr.Button("📏 Intervalles égaux")
                    bin_quantile_btn = gr.Button("📊 Intervalles par quantiles")
            
            norm_output = gr.Dataframe(label="Jeu de données après normalisation")
            status_norm = gr.Textbox(label="Statut")
        
        # =============================
        # 🏷️ ENCODAGE
        # =============================
        with gr.Accordion("🏷️ Encodage Avancé", open=False):
            gr.Markdown("### Encoder des colonnes catégorielles")
            with gr.Row():
                encode_cols = gr.Dropdown(
                label="Colonne",
                choices=[],
                interactive=True
                )
                dataset_state.change(
                    fn=get_columns,
                    inputs=dataset_state,
                    outputs=encode_cols
                )
                with gr.Column():
                    encode_label_btn = gr.Button("🔢 Encodage par étiquettes")
                    encode_onehot_btn = gr.Button("🎯 Encodage One-Hot")
                    encode_binary_btn = gr.Button("⚖️ Encodage binaire")
            
            encode_output = gr.Dataframe(label="Jeu de données après encodage")
            status_encode = gr.Textbox(label="Statut")
        
        


        # =============================
        # 🔗 CONNEXIONS DES BOUTONS
        # =============================
        
        delete_col_btn.click(
            delete_column, 
            inputs=[dataset_state, col_to_delete], 
            outputs=[delete_output, dataset_state, status_delete]
        )
        
        
        delete_outliers_btn.click(
            delete_rows_outliers_iqr,
            inputs=[dataset_state, col_outliers],
            outputs=[delete_output, dataset_state, status_delete]
        )
        
        delete_dup_btn.click(
            delete_duplicates,
            inputs=dataset_state,
            outputs=[delete_output, dataset_state, status_delete]
        )
        
        delete_nan_btn.click(
            delete_missing_rows,
            inputs=dataset_state,
            outputs=[delete_output, dataset_state, status_delete]
        )
        
        replace_custom_btn.click(
            replace_value_custom,
            inputs=[dataset_state, col_replace, old_val, new_val],
            outputs=[replace_output, dataset_state, status_replace]
        )
        
        replace_mean_btn.click(
            replace_value_with_mean,
            inputs=[dataset_state, col_auto, value_to_replace],
            outputs=[replace_output, dataset_state, status_replace]
        )
        
        replace_median_btn.click(
            replace_value_with_median,
            inputs=[dataset_state, col_auto, value_to_replace],
            outputs=[replace_output, dataset_state, status_replace]
        )
        
        replace_mode_btn.click(
            replace_value_with_mode,
            inputs=[dataset_state, col_auto, value_to_replace],
            outputs=[replace_output, dataset_state, status_replace]
        )
        
        replace_outliers_iqr_btn.click(
            replace_outliers_with_iqr,
            inputs=[dataset_state, col_auto],
            outputs=[replace_output, dataset_state, status_replace]
        )
        
        fill_mean_btn.click(
            lambda df, col: fill_missing_with_strategy(df, col, 'mean'),
            inputs=[dataset_state, col_fill],
            outputs=[fill_output, dataset_state, status_fill]
        )
        
        fill_median_btn.click(
            lambda df, col: fill_missing_with_strategy(df, col, 'median'),
            inputs=[dataset_state, col_fill],
            outputs=[fill_output, dataset_state, status_fill]
        )
        
        fill_mode_btn.click(
            lambda df, col: fill_missing_with_strategy(df, col, 'mode'),
            inputs=[dataset_state, col_fill],
            outputs=[fill_output, dataset_state, status_fill]
        )
        
        fill_zero_btn.click(
            lambda df, col: fill_missing_with_strategy(df, col, 'zero'),
            inputs=[dataset_state, col_fill],
            outputs=[fill_output, dataset_state, status_fill]
        )
    
        norm_standard_btn.click(
            lambda df, cols: normalize_columns(df, cols, 'standard'),
            inputs=[dataset_state, norm_cols],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        norm_minmax_btn.click(
            lambda df, cols: normalize_columns(df, cols, 'minmax'),
            inputs=[dataset_state, norm_cols],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        norm_robust_btn.click(
            lambda df, cols: normalize_columns(df, cols, 'robust'),
            inputs=[dataset_state, norm_cols],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        norm_log_btn.click(
            lambda df, cols: normalize_columns(df, cols, 'log'),
            inputs=[dataset_state, norm_cols],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        bin_equal_btn.click(
            lambda df, col, n: create_bins(df, col, n, 'equal'),
            inputs=[dataset_state, bin_col, n_bins],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        bin_quantile_btn.click(
            lambda df, col, n: create_bins(df, col, n, 'quantile'),
            inputs=[dataset_state, bin_col, n_bins],
            outputs=[norm_output, dataset_state, status_norm]
        )
        
        encode_label_btn.click(
            lambda df, cols: encode_multiple_columns(df, cols, 'label'),
            inputs=[dataset_state, encode_cols],
            outputs=[encode_output, dataset_state, status_encode]
        )
        
        encode_onehot_btn.click(
            lambda df, cols: encode_multiple_columns(df, cols, 'onehot'),
            inputs=[dataset_state, encode_cols],
            outputs=[encode_output, dataset_state, status_encode]
        )
        
        encode_binary_btn.click(
            lambda df, cols: encode_multiple_columns(df, cols, 'binary'),
            inputs=[dataset_state, encode_cols],
            outputs=[encode_output, dataset_state, status_encode]
        )

        return dataset_state
    
