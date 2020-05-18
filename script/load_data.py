import csv
import pandas as pd
import numpy as np
from IPython.display import display


def load_data():
    pd.set_option('display.max_columns', None)
    pd.set_option("max_rows", 30)

    df = pd.read_csv('data/testset.csv', sep=';', engine='python')

    df = df.drop(columns=['prob_REF', 'prediction', 'Unnamed: 0', 'Data valutazione pratica'])

    df.drop_duplicates(subset='sCodiceDomandaSiel', inplace=True)

    df.columns = ['rischio totale', 'domanda finanziamento', 'outstanding',
                  'totale finanziato gruppo', 'totale finanziato',
                  'rating bplg', 'rating bnp',
                  'assilea', 'cerved', 'nuovo cliente',
                  'nuovo gruppo', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'Z17',
                  'BLS', 'ELS', 'RD', 'TS', 'Cessione Contratto Mandatata',
                  'Cessione Contratto Semplice', 'Credito',
                  'Enveloppe loc, opérations Spé', 'Enveloppe location Transfert',
                  'Enveloppe location immobilier', 'Locaz. Finanziaria Immobiliare',
                  'Locazione Finanziaria', 'Locazione Operativa', 'Package credito',
                  'Package locazione', 'Riacquisto di crediti',
                  'target', 'codice', 'prob']

    df = df.reindex(columns=['codice', 'rischio totale', 'domanda finanziamento', 'outstanding',
                        'totale finanziato gruppo', 'totale finanziato',
                        'rating bplg', 'rating bnp',
                        'assilea', 'cerved', 'nuovo cliente',
                        'nuovo gruppo', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'Z17',
                        'BLS', 'ELS', 'RD', 'TS', 'Cessione Contratto Mandatata',
                        'Cessione Contratto Semplice', 'Credito',
                        'Enveloppe loc, opérations Spé', 'Enveloppe location Transfert',
                        'Enveloppe location immobilier', 'Locaz. Finanziaria Immobiliare',
                        'Locazione Finanziaria', 'Locazione Operativa', 'Package credito',
                        'Package locazione', 'Riacquisto di crediti',
                        'prob', 'target'])

    df['target'].replace(to_replace={'ACC' : 1, 'REF' : 0}, inplace=True)

    df = clean_data(df)
    #display(df)
    return df

def clean_data(df):
    df = df.drop(columns=['codice', 'prob'])
    df['rating bplg'].replace(to_replace={0: np.nan}, inplace=True)
    df['nuovo cliente'] = df['nuovo cliente'].notnull().astype('int')
    df['nuovo gruppo'] = df['nuovo gruppo'].notnull().astype('int')
    return df
