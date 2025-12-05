import pandas as pd
import os
from pathlib import Path

# Obtenir le chemin du script actuel
script_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin du dossier contenant les fichiers CSV (dans le même répertoire que le script)
folder_path = os.path.join(script_dir, 'extracted_csvs')

# Liste pour stocker tous les DataFrames
all_dataframes = []

# Compteurs
total_files = 0
successful_files = 0
failed_files = 0

print("=== FUSION DES FICHIERS CSV ===\n")

# Parcourir tous les fichiers CSV dans le dossier
for file_name in os.listdir(folder_path):
    if file_name.endswith('_workers.csv'):
        total_files += 1
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Lire chaque fichier CSV
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
            successful_files += 1
            
            if total_files % 50 == 0:  # Afficher la progression tous les 50 fichiers
                print(f"✓ {total_files} fichiers traités...")
        
        except Exception as e:
            failed_files += 1
            print(f"✗ Erreur avec {file_name}: {e}")

# Fusionner tous les DataFrames
if all_dataframes:
    print(f"\n=== RÉSULTATS ===")
    print(f"Fichiers trouvés : {total_files}")
    print(f"Fichiers chargés avec succès : {successful_files}")
    print(f"Fichiers échoués : {failed_files}")
    
    # Concaténer tous les DataFrames
    merged_dataset = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n=== DONNÉES FUSIONNÉES ===")
    print(f"Nombre total de lignes : {len(merged_dataset)}")
    print(f"Nombre de colonnes : {len(merged_dataset.columns)}")
    print(f"Colonnes : {list(merged_dataset.columns)}")
    
    # Informations sur les données
    print(f"\n=== STATISTIQUES ===")
    print(f"ID_adherent uniques : {merged_dataset['ID_adherent'].nunique()}")
    print(f"Mois uniques : {merged_dataset['month'].nunique()}")
    print(f"\nAperçu des données :")
    print(merged_dataset.head(10))
    
    # Sauvegarder le fichier fusionné
    output_file = os.path.join(script_dir, 'merged_salaries.csv')
    merged_dataset.to_csv(output_file, index=False)
    print(f"\n✓ Dataset fusionné sauvegardé dans '{output_file}'")
    
else:
    print("⚠️ Aucun fichier CSV trouvé dans le dossier")
