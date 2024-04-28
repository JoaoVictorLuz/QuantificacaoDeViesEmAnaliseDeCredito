import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import ast
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Fairness lib
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import EqOddsPostprocessing

def justica_exps(base, attr_set, df, parametros, modelo, categorical_features, label, fvr_classes, prt_attrs, priv_classes, unprivileged_groups, privileged_groups, unprivileged_groups_fe, privileged_groups_fe, ):

  # criando data set no formato adequado (classe Standard Dataset)
  df2model = StandardDataset(df,
                            label_name=label,
                            favorable_classes=fvr_classes,
                            protected_attribute_names=prt_attrs,
                            privileged_classes=priv_classes)

  i = 0
  results = []
  for param_set in parametros:
    result_grid = {}
    print(type(param_set))
    print(param_set)

    # Estabelecendo conjunto de parametros para o modelo
    # Criando modelo
    if modelo == 'gbt':
      clf = GradientBoostingClassifier(n_estimators = param_set['n_estimators'],
                                        min_samples_split = param_set['min_samples_split'],
                                        learning_rate = param_set['learning_rate'],
                                        max_depth = param_set['max_depth'],
                                        random_state = param_set['random_state'])
    elif modelo == 'rf':
      clf = RandomForestClassifier(n_estimators = param_set['n_estimators'],
                                  min_samples_split = param_set['min_samples_split'],
                                  criterion = param_set['criterion'],
                                  max_depth = param_set['max_depth'],
                                  random_state = param_set['random_state'])
    else:
      print("Modelos suportados atualmente: 'gbt' e 'rf'. ")

    # Separando teste e treino
    df2_train, df2_test = df2model.split([0.7], shuffle=True)

    # Fitando o modelo na base de treino
    model = clf.fit(df2_train.features, df2_train.labels.ravel())

    # Formatando base de teste
    x_df2_test = df2_test.features
    y_df2_test = df2_test.labels.ravel()

    # Criando copia da base para predição
    dataset = df2_test
    dataset_pred = dataset.copy()
    dataset_pred.labels = model.predict(df2_test.features)

    # # Definindo as populações privilegiadas e não-privilegiadas
    # # Sexo
    # ## Recuperando o indice
    # privileged_sex   = np.where(categorical_names['Sexo'] == 'male')[0]
    # unprivileged_sex = np.where(categorical_names['Sexo'] == 'female')[0]
    # ## Estabelecendo formato, seguindo o formato da documentacao das classes ClassificationMetric and BinaryLabelDatasetMetric
    # privileged_groups   = [{'Sexo' : privileged_sex}]
    # unprivileged_groups = [{'Sexo' : unprivileged_sex}]

    # Criando classes de populacoes
    classified_metric_sex = ClassificationMetric(dataset,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    metric_pred_sex = BinaryLabelDatasetMetric(dataset_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

    # # Idade
    # ## Recuperando o indice
    # privileged_faixa_etaria   = np.where(categorical_names['faixa_etaria'] == 'adulto')[0]
    # unprivileged_faixa_etaria = np.where(categorical_names['faixa_etaria'] == 'jovem')[0]
    # ## Estabelecendo formato, seguindo o formato da documentacao das classes ClassificationMetric and BinaryLabelDatasetMetric
    # privileged_groups_fe   = [{'faixa_etaria' : privileged_faixa_etaria}]
    # unprivileged_groups_fe = [{'faixa_etaria' : unprivileged_faixa_etaria}]

    # Criando classes de populacoes
    classified_metric_faixa_etaria = ClassificationMetric(dataset,
                                                          dataset_pred,
                                                          unprivileged_groups=unprivileged_groups_fe,
                                                          privileged_groups=privileged_groups_fe)

    metric_pred_faixa_etaria = BinaryLabelDatasetMetric(dataset_pred,
                                                        unprivileged_groups=unprivileged_groups_fe,
                                                        privileged_groups=privileged_groups_fe)

    # Submetendo modelo ao 10-fold cross validation (fitting)
    # obtendo acurácia
    scores1 = cross_val_score(clf, df2_train.features, df2_train.labels.ravel(), cv=10)
    print("%0.2f de acuracia com desvio padrão de %0.2f" % (scores1.mean(), scores1.std()))
    # obtendo f1_score
    scores2 = cross_val_score(clf, df2_train.features, df2_train.labels.ravel(), cv=10,scoring='f1_macro')
    print("%0.2f de f1 com desvio padrão de %0.2f" % (scores2.mean(), scores2.std()))
    # obtendo precisão
    scores3 = cross_val_score(clf, df2_train.features, df2_train.labels.ravel(), cv=10,scoring='precision')
    print("%0.2f de precisão com desvio padrão de %0.2f" % (scores3.mean(), scores3.std()))

    # Obtendo o score no teste
    model = clf.fit(df2_train.features, df2_train.labels.ravel())
    score4 = model.score(x_df2_test, y_df2_test)
    print("%0.2f de score no teste" % (score4))

    # Métricas de justica algoritmica
    print('STATISTICAL PARITY DIFFERENCE')
    spd_sexo = metric_pred_sex.statistical_parity_difference()
    spd_faixa_etaria = metric_pred_faixa_etaria.statistical_parity_difference()

    print(f'Atributo sexo: {spd_sexo}')
    print(f'Atributo faixa_etaria: {spd_faixa_etaria}')

    print('')

    print('DISPARATE IMPACT')
    di_sexo = metric_pred_sex.disparate_impact()
    di_faixa_etaria = metric_pred_faixa_etaria.disparate_impact()

    print(f'Atributo sexo: {di_sexo}')
    print(f'Atributo faixa_etaria: {di_faixa_etaria}')

    print(' ')

    print('AVERAGE ODDS DIFFERENCE')
    aod_sexo = classified_metric_sex.average_odds_difference()
    aod_faixa_etaria = classified_metric_faixa_etaria.average_odds_difference()

    print(f'Atributo sexo: {aod_sexo}')
    print(f'Atributo faixa_etaria: {aod_faixa_etaria}')

    print(' ')

    print('EQUAL OPPORTUNITY DIFFERENCE')

    eod_sexo = classified_metric_sex.equal_opportunity_difference()
    eod_faixa_etaria = classified_metric_faixa_etaria.equal_opportunity_difference()
    print(f'Atributo sexo: {eod_sexo}')
    print(f'Atributo faixa_etaria: {eod_faixa_etaria}')

    # collecting results
    result_grid['index'] = i
    result_grid['base'] = base
    result_grid['attr_set'] = attr_set
    result_grid['modelo'] = modelo
    result_grid['param_set'] = str(param_set)
    result_grid['10fold-acuracia'] = scores1.mean()
    result_grid['10fold-f1'] = scores2.mean()
    result_grid['10fold-precisao'] = scores3.mean()
    result_grid['test-score'] = score4
    result_grid['spd_sexo'] = spd_sexo
    result_grid['spd_faixa_etaria'] = spd_faixa_etaria
    result_grid['di_sexo'] = di_sexo
    result_grid['di_faixa_etaria'] = di_faixa_etaria
    result_grid['aod_sexo'] = aod_sexo
    result_grid['aod_faixa_etaria'] = aod_faixa_etaria
    result_grid['eod_sexo'] = eod_sexo
    result_grid['eod_faixa_etaria'] = eod_faixa_etaria

    results.append(result_grid)
    i += 1
    print(result_grid)
    print("")
  return result_grid

def preparando_experimentos():
  np.random.seed(2024)

  list_gbt_params = []
  for estimators in range(100, 500, 100):
    for samples in range(50, 300, 50):
      for learning in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for depth in range(2, 8, 1):
          gbt_p = {}
          RANDOM_STATE = 2024
          gbt_p['n_estimators'] = estimators
          gbt_p['min_samples_split'] = samples
          gbt_p['learning_rate'] = learning
          gbt_p['max_depth'] = depth
          gbt_p['random_state'] = RANDOM_STATE
          list_gbt_params.append(gbt_p)

  # print(*(x for x in list_gbt_params), sep='\n')
  print("Total de {} conjuntos de parâmetros.".format(len(list_gbt_params)))

  list_rf_params = []
  for estimators in range(100, 500, 100):
    for samples in range(50, 300, 50):
      for learning in ['gini', 'entropy']:
        for depth in range(2, 8, 1):
          rf_p = {}
          RANDOM_STATE = 2024
          rf_p['n_estimators'] = estimators
          rf_p['min_samples_split'] = samples
          rf_p['criterion'] = learning
          rf_p['max_depth'] = depth
          rf_p['random_state'] = RANDOM_STATE
          list_rf_params.append(rf_p)

  # print(*(x for x in list__params), sep='\n')
  print("Total de {} conjuntos de parâmetros.".format(len(list_rf_params)))

  return list_gbt_params 



