import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython.display import display

def missing_values(df):

    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    display(pd.DataFrame({'Missing Ratio': all_data_na}))


def pca_compute(df):
    dfd = df.dropna(axis=0, how='any', thresh=None, subset=['Assilea'], inplace=False)
    x = dfd.loc[:, ['rischio totale', 'domanda finanziamento', 'outstanding',
                    'totale finanziato gruppo', 'totale finanziato',
                    'rating bplg', 'rating bnp',
                    'Assilea']].values
    y = dfd.loc[:, ['target']].values

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(x)
    pca_df = pd.DataFrame(data=new_data,
                          columns=['principal component 1', 'principal component 2'])
    pca_df_complete = pca_df
    pca_df_complete['target'] = df[['target']]

    plt.figure(figsize=(8, 8))
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.title('Principal Component Analysis (2 Components)', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = pca_df_complete['target'] == target
        plt.scatter(pca_df_complete.loc[indicesToKeep, 'principal component 1']
                    , pca_df_complete.loc[indicesToKeep, 'principal component 2']
                    , c=color
                    , s=10)
    # ax.legend(targets)
    plt.axis([-3, 3, -3, 3])
    plt.grid()

    print("Explained Variance")
    print("   Component 1 %3.2f" % (pca.explained_variance_ratio_[0]))
    print("   Component 2 %3.2f" % (pca.explained_variance_ratio_[1]))
    print("   Total Explained Variance %3.2f" % sum(pca.explained_variance_ratio_))

def visualize(df):
    missing_values(df)
    pca_compute(df)