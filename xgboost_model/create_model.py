from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
import optuna
import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import numpy as np
import json
import datetime
import warnings
from pathlib import Path
import joblib,pickle
import plotly.express as px
import matplotlib.pyplot as plt
# Ignore multiple warning categories
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)


metric_test = 'MAE'

def create_ferie(df):
    df.date = pd.to_datetime(df.date)
    year_max = df.date.max().year
    year_min = df.date.min().year
    label = LabelEncoder()
    dico = {}

    for k,v in holidays.country_holidays('FR',years=range(year_min, year_max+1),language='fr').items() :
        dico[k]=v

    df['ferie']=df.date.map(dico)
    df.ferie = df.ferie.fillna('0')
    eptica.ferie = LabelEncoder().fit_transform(eptica.ferie)
    df.sort_values(by='date',inplace=True)

    return df

def create_best_model(df):

    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1,log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.1, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
        lags = trial.suggest_int('lags', 35, 364, step=7) # step means we only try multiples of 7 starting from 14
        models = [XGBRegressor(random_state=0, n_estimators=500,    learning_rate=learning_rate, max_depth=max_depth,
                            min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree)]


        model = MLForecast(models=models,
                        freq='D',
                        lags=[1,7,14,21,28, lags],
                    #   lag_transforms={
                    #     1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                    # }
                        date_features=['dayofweek', 'month','weekday'],

                        num_threads=6)


        model.fit(data_train, id_col='entite', time_col='date', target_col='recus', static_features=['ferie'])
        h=data_test.date.nunique()
        df_pred = model.predict(h= h,dynamic_dfs=[data_test[['entite','date']+['ferie']]])
        df_pred = df_pred.merge(data_test[['entite', 'date', 'recus']], on=['entite', 'date'], how='left')
        trial.set_user_attr(key="best_booster", value=model)

        if metric_test == 'RMSE':
            error = mean_squared_error(df_pred['recus'], df_pred['XGBRegressor'],squared=False)
        else:
            error = mean_absolute_error(df_pred['recus'], df_pred['XGBRegressor'])

        return error
    data_train = df[df.date < '2022-03-15']
    data_test = df[(df.date >= '2022-03-15') & (df.date < '2023-01-01')]
    models = XGBRegressor(random_state=0, n_estimators=100)
    model = MLForecast(models=models,
                    freq='D',
                    lags=[1,2,3,4,5,6,7,28,60,180,364],
                    date_features=['year', 'weekday', 'month', 'dayofyear'],
                    num_threads=6)

    model.fit(data_train, id_col= 'entite' , time_col='date', target_col='recus',static_features=['ferie'])
    h=data_test.date.nunique()

    df_pred = model.predict(h = h , dynamic_dfs=[data_test[['entite','date']+['ferie']]])
    df_pred = df_pred.merge(data_test[['entite', 'date', 'recus']], on=['entite', 'date'], how='left')
    df_pred.rename(columns={'XGBRegressor':'pred'},inplace=True)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30,callbacks=[callback]) #
    ##le choix du trial dépend de vous 20 permets de trouver les hyperparamètres à un modèle qui est assez bon mais vous pouvez essayez de choisir beaucoup plus de trials si
    # si vous voulez de meilleurs résultat n'hésitez pas à changer les suggests pour tester un large panel de paramètres la fonction objective

    best_model=study.user_attrs["best_booster"] 
    return best_model,study
  
def mega_test(df,model):

    def metrics(df_pred,model,i):

        date = datetime.date.today()
        path= f"/home/apprenant/Bureau/Projet_CH/xgboost_model/resultat/{metric_test}/resultat_{date}/resultat_Trainset_2022_+{i}_Mois"
        file_path = Path(path)
        if not file_path.exists():
            file_path.mkdir()
            print(f"Répertoire '{file_path}' créé avec succès.")
        else :
            print('Le Répertoire trainset existe déja')
        temp_list = []
        temp_rp = []
        dico = {}
        dico['MAPE GLOBAL'] = str(round(mean_absolute_percentage_error(df_pred['recus'],df_pred['pred'])*100)) +' %'
        dico['MAE GLOBAL'] = str(round(mean_absolute_error(df_pred['recus'],df_pred['pred']))) +' Mails'
        dico['RP GLOBAL'] = str(round(((df_pred['recus']/df_pred['pred'])*100).mean())) +' %'
        dico['Features']=pd.Series(model.models_['XGBRegressor'].feature_importances_, index=model.ts.features_order_).sort_values(ascending=False)
        # dico['Shap_values']=shap_values
        for entite in df_pred.entite.unique():
            new_df = df_pred[df_pred.entite == entite]
            dico['MAPE pour {}'.format(entite)] = ': {} %'.format(round(mean_absolute_percentage_error(new_df['recus'],new_df['pred'])*100))
            dico['MAE pour {}'.format(entite)] = ': {} Mails'.format(round(mean_absolute_error(new_df['recus'],new_df['pred'])))
            dico['RP pour {}'.format(entite)]  = ': {} %'.format(round(((new_df['recus']/new_df['pred'])*100).mean()))
            temp_list.append(round(mean_absolute_percentage_error(new_df['recus'],new_df['pred'])*100))
            temp_rp.append(round(((new_df['recus']/new_df['pred'])*100).mean()))

            train_path = Path(path + f'/resultat_{entite}_Trainset_2022_+{i}_Mois.png')
            
            # print(new_df)
            fig, ax = plt.subplots()
            # Tracer les données
            ax.plot(new_df['date'], new_df['recus'], label='Réel')
            ax.plot(new_df['date'], new_df['pred'], label='Prédiction')
            # Ajouter une légende
            ax.set_title(f'Répartition des appels de {entite}')
            ax.legend()
            # Définir les labels des axes
            ax.set_xlabel('Date')
            ax.set_ylabel('Mails recus')
            # Définir la taille de la figure
            fig.set_size_inches(8, 5)
            # Sauvegarder la figure dans un dossier spécifié
            fig.savefig(train_path, dpi=300) 
            plt.close()





        idx = temp_list.index(min(temp_list))
        idx2 = temp_list.index(max(temp_list))
        idx3 = temp_rp.index(min(temp_rp))
        idx4 = temp_rp.index(max(temp_rp))


        print(f'La MAPE la plus faible est {temp_list[idx]} %'+ f"de l'entité {df_pred.entite.unique()[idx]}")
        print(f'La MAPE la plus haute est {temp_list[idx2]} %'+ f"de l'entité {df_pred.entite.unique()[idx2]}")
        print(f'Le RP le plus faible est {temp_rp[idx3]} '+ f"de l'entité {df_pred.entite.unique()[idx3]}")
        print(f'Le RP le plus haute est {temp_rp[idx4]} '+ f"de l'entité {df_pred.entite.unique()[idx4]}")
        print('#############################################################################################')

        # json_str = json.dumps(str(dico), indent=4)  
        with open(path + f'/resultat_Trainset_2022_+{i}_Mois.txt' ,"w") as file:
    # Iterate through dictionary items and write them to the file
            for key, value in dico.items():
                file.write(f"{key}: {value}\n")

        return dico
    
    def regle_ferie(df):
        liste_ferie = list(holidays.country_holidays('FR',years=range(2020, 2024),language='fr').keys())
        for i in range(0,(len(df))):
            if df.loc[i].date in liste_ferie :
                if df.loc[i].date.weekday() in [1,2,3,4]:
                    try :
                        df.loc[i, 'pred'] = df.loc[i-1 , 'pred']*0.3
                    except :
                        print(df.loc[i,'date'])
                elif df.loc[i].date.weekday() == 0 :
                    try :
                        df.loc[i, 'pred'] = df.loc[i-3 , 'pred']*0.3
                    except :
                        print(df.loc[i,'date'])
        return df
    register = {}
    data_train = df[df.date < '2023-01-01']
    data_test = df[df.date > '2022-12-31']
    best_params = None
    # m = data_test.date.max().month
    dicoframe ={}
    for i in range(0,8):

        if i ==0 :
            print(i)
            model.fit(data_train, id_col= 'entite' , time_col='date', target_col='recus',static_features=['ferie'])
            h=data_test.date.nunique()

            df_pred = model.predict(horizon = 30 , dynamic_dfs=[data_test[['entite','date']+['ferie']]])
            df_pred = df_pred.merge(data_test[['entite', 'date', 'recus']], on=['entite', 'date'], how='left')
            df_pred.rename(columns={'XGBRegressor':'pred'},inplace=True)

            # explainer = shap.TreeExplainer(model)
            # shap_values = explainer.shap_values(data_test)
            dicoframe[f'df{i}']=df_pred
            register['Résultat avec Train_set 2022 M+{}'.format(i)]=metrics(df_pred,model,i)

        else :
            print('Trainset 2022 + '+str(i) +'Mois')

            df_next = data_test[(data_test.date > pd.to_datetime('2022-12-31')) & (data_test.date < pd.to_datetime('2023-{}-01'.format(i+1)))]
            data_test2 = data_test[(data_test.date >= pd.to_datetime('2023-{}-01'.format(i+1))) & (data_test.date < pd.to_datetime('2023-{}-01'.format(i+2)))]
            data_train = pd.concat([data_train,df_next])

            # print(data_test)

            model.fit(data_train, id_col= 'entite' , time_col='date', target_col='recus',static_features=['ferie'])
            h=data_test2.date.nunique()
            # print(len(data_train))
            df_pred = model.predict(horizon = h , dynamic_dfs=[data_test2[['entite','date']+['ferie']]])
            df_pred = df_pred.merge(data_test[['entite', 'date', 'recus']], on=['entite', 'date'], how='left')
            df_pred.rename(columns={'XGBRegressor':'pred'},inplace=True)

            # explainer = shap.TreeExplainer(model)
            # shap_values = explainer.shap_values(data_test)
            df_pred = regle_ferie(df_pred)
            tmp = df_pred.select_dtypes(include=[np.number])
            df_pred.loc[:, tmp.columns] = np.round(tmp)
            dicoframe[f'df{i}']=df_pred
            register['Résultat avec Train_set 2022 M+{}'.format(i)]=metrics(df_pred,model,i)

    return register,dicoframe


# Spécifiez le nom du répertoire que vous souhaitez créer
date = datetime.date.today()

nom_repertoire = f'resultat_{date}'

# Créez un objet Path pour le répertoire
file_path_metric =  f"/home/apprenant/Bureau/Projet_CH/xgboost_model/resultat/{metric_test}"
chemin_metric = Path(file_path_metric)
if not chemin_metric.exists():
    # Créez le répertoire
    chemin_metric.mkdir()
    print(f"Répertoire '{metric_test}' créé avec succès.")
else:
    print(f"Le répertoire '{metric_test}' existe déjà.")


file_path = f"/home/apprenant/Bureau/Projet_CH/xgboost_model/resultat/{metric_test}/{nom_repertoire}"
file_path_model = file_path+'/model'
chemin_repertoire = Path(file_path)
chemin_model=Path(file_path_model)
# Vérifiez si le répertoire existe déjà
if not chemin_repertoire.exists():
    # Créez le répertoire
    chemin_repertoire.mkdir()
    print(f"Répertoire '{nom_repertoire}' créé avec succès.")
else:
    print(f"Le répertoire '{nom_repertoire}' existe déjà.")

if not chemin_model.exists():
    # Créez le répertoire
    chemin_model.mkdir()
    print("Répertoire model créé avec succès.")
else:
    print("Le répertoire model existe déjà.")


eptica = pd.read_csv('/home/apprenant/Bureau/Projet_CH/data/eptica_pro_format.csv')

eptica.columns = ['date','entite','instance', 'recus']
eptica = eptica.groupby(['date','entite']).sum(numeric_only=True).reset_index()


eptica = create_ferie(eptica)

print("""
        ########################################################################################
        #                                                                                      #
        #                                                                                      #
        #          Création du modèle avec les meilleurs hyperparamètres en cours...           #
        #                                                                                      #
        #                                                                                      #
        ######################################################################################## """)


best_model,study = create_best_model(eptica)

best_params= json.dumps(study.best_params, indent=4)  # indent for pretty formatting

with open(file_path+'/model/best_params.txt', "w") as file:
    file.write(best_params)

best_value = json.dumps(study.best_trial.values, indent=4) 
with open(file_path+'/model/best_result.txt', "w") as file:
    if metric_test == 'RMSE':
        file.write('RMSE : ' +best_value)
    else:
        file.write('MAE : ' +best_value)


with open(file_path+f'/model/best_model_{date}.pkl', 'wb') as f:
    pickle.dump(best_model.models['XGBRegressor'], f)


feature_importances = pd.Series(best_model.models_['XGBRegressor'].feature_importances_, index=best_model.ts.features_order_).sort_values(ascending=False)
fig, ax = plt.subplots()

# Tracer le graphique à barres
feature_importances.plot.bar(ax=ax, title='Feature Importance XGBRegressor')

# Définir les labels des axes
ax.set_xlabel('Features')
ax.set_ylabel('Importance')

# Définir la taille de la figure
fig.set_size_inches(8, 5)
fig.savefig(file_path+'/model/features_importance.png', dpi=300)  




print("""
        ########################################################################################
        #                                                                                      #
        #                                                                                      #
        #          Test en cours...                                                            #
        #                                                                                      #
        #                                                                                      #
        ######################################################################################## """)

result,dicoframe = mega_test(eptica,best_model)


