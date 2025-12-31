import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploration_results(df):
    """
    Exploration complète du dataset avec graphiques matplotlib
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None, None, None, None, None, None, None, None, None

    try:
        # 1. Aperçu (head)
        head_df = df.head(10)

        # 2. Overview (dimensions + doublons)
        overview_data = {
            "Métrique": ["Nombre de lignes", "Nombre de colonnes", "Doublons"],
            "Valeur": [df.shape[0], df.shape[1], df.duplicated().sum()]
        }
        overview_df = pd.DataFrame(overview_data)

        # 3. Types de données
        dtypes_data = {
            "Colonne": df.columns.tolist(),
            "Type": df.dtypes.astype(str).tolist()
        }
        dtypes_df = pd.DataFrame(dtypes_data)

        # 4. Valeurs manquantes
        missing_data = {
            "Colonne": df.columns.tolist(),
            "Manquantes": df.isnull().sum().tolist(),
            "Pourcentage": (df.isnull().sum() / len(df) * 100).round(2).tolist()
        }
        missing_df = pd.DataFrame(missing_data)

        # 5. Valeurs uniques
        unique_data = {
            "Colonne": df.columns.tolist(),
            "Uniques": [df[col].nunique() for col in df.columns]
        }
        unique_df = pd.DataFrame(unique_data)

        # 6. Statistiques descriptives
        stats_df = df.describe(include='all').T.reset_index()
        stats_df.columns = ['Colonne'] + stats_df.columns[1:].tolist()

        # 7. HISTOGRAMMES
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            # Augmenter la taille pour mieux voir
            fig_hist, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            fig_hist.suptitle('📊 Distribution des variables numériques', 
                             fontsize=16, fontweight='bold', y=0.995)
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', 
                              edgecolor='black', alpha=0.7)
                axes[idx].set_title(col, fontweight='bold')
                axes[idx].set_xlabel('Valeur')
                axes[idx].set_ylabel('Fréquence')
                axes[idx].grid(True, alpha=0.3)
            
            # Masquer les axes vides
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
        else:
            fig_hist = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Aucune colonne numérique", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')

        # 8. BOXPLOTS
        if numeric_cols:
            n_cols_box = min(3, len(numeric_cols))
            n_rows_box = (len(numeric_cols) + n_cols_box - 1) // n_cols_box
            
            fig_box, axes_box = plt.subplots(n_rows_box, n_cols_box, 
                                            figsize=(15, 5*n_rows_box))
            fig_box.suptitle('📦 Boxplots des variables numériques', 
                            fontsize=16, fontweight='bold', y=0.995)
            
            if n_rows_box == 1 and n_cols_box == 1:
                axes_box = [axes_box]
            else:
                axes_box = axes_box.flatten() if n_rows_box > 1 or n_cols_box > 1 else [axes_box]
            
            for idx, col in enumerate(numeric_cols):
                axes_box[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                     boxprops=dict(facecolor='lightseagreen', alpha=0.7),
                                     medianprops=dict(color='red', linewidth=2))
                axes_box[idx].set_title(col, fontweight='bold')
                axes_box[idx].set_ylabel('Valeur')
                axes_box[idx].grid(True, alpha=0.3, axis='y')
            
            # Masquer les axes vides
            for idx in range(len(numeric_cols), len(axes_box)):
                axes_box[idx].axis('off')
            
            plt.tight_layout()
        else:
            fig_box = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Aucune colonne numérique", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')

        # 9. MATRICE DE CORRÉLATION
        if len(numeric_cols) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
            corr_matrix = df[numeric_cols].corr()
            
            # Utiliser seaborn pour une meilleure visualisation
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, square=True, linewidths=1, 
                       cbar_kws={"shrink": 0.8}, ax=ax_corr,
                       vmin=-1, vmax=1)
            
            ax_corr.set_title('🔗 Matrice de Corrélation (Pearson)', 
                            fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
        else:
            fig_corr = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Pas assez de variables numériques\npour la corrélation", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')

        return (
            head_df,
            overview_df,
            dtypes_df,
            missing_df,
            unique_df,
            stats_df,
            fig_hist,
            fig_box,
            fig_corr
        )

    except Exception as e:
        print(f"Erreur dans exploration_results: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None