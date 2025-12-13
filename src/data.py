import pandas as pd
import glob
import os


def load_data(path_to_datasets, n_assets_per_sector=5):
    """
    Charge les fichiers CSV, extrait les prix et fusionne en un seul DataFrame.
    """
    all_files = glob.glob(os.path.join(path_to_datasets, "*.csv"))

    price_data = pd.DataFrame()

    print(f"Fichiers trouvés : {len(all_files)}")

    for filename in all_files:
        sector_name = os.path.basename(filename).replace(".csv", "")
        # Lecture du CSV
        df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')

        # NOTE : Adaptez 'Adj Close' si vos colonnes s'appellent autrement (ex: 'Close', 'prix', etc.)
        # On suppose ici que le CSV a des colonnes par ticker.
        # Si le format est différent (ex: colonne 'Ticker'), il faudra adapter ce bloc.

        # Pour l'exemple, prenons les N premiers actifs s'il y a beaucoup de colonnes
        assets_to_take = df.columns[:n_assets_per_sector]
        df_sector = df[assets_to_take]

        if price_data.empty:
            price_data = df_sector
        else:
            price_data = pd.concat([price_data, df_sector], axis=1)

    # Nettoyage : suppression des lignes avec des valeurs manquantes
    price_data.dropna(inplace=True)
    return price_data