# **III. SUPERVISED LEARNING**
models_result = pd.DataFrame()

def models(X, y):
    global models_result
    """
    Функция 
    - трансформирует данные: применяет StandardScaler к числовым данным и кодирует категориальные данные;
    - применяет GridSearchCV для тюнига гиперпараметров
    - оценивает модели на валидационных данных и записывает оценки в df
    - демонстрирует полученные оценки по каждой модели и показывает важность признаков через permutation_importance plot
    - возвращает 4 модели: RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, VotingClassifier
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

    # ######### preprocessing ##########################################
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler",StandardScaler())] 
        )

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),('encoder',OneHotEncoder(handle_unknown="ignore"))]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ]
    )
    # ######### RandomForestClassifier #################################

    pipe_RFC = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))] 
    )

    parameter_grid = {'classifier__criterion': [None],
                        'classifier__min_samples_split':[sp_randint.rvs(2,14,2)],
                        'classifier__min_samples_leaf': [sp_randint.rvs(10,16,2)],
                        'classifier__n_estimators' : [sp_randint.rvs(500,600,50)],
                        'classifier__criterion': ["gini", "entropy"]}

    gsRFC = GridSearchCV(estimator=pipe_RFC, param_grid = parameter_grid, cv=K_fold, scoring="accuracy", n_jobs= -1, verbose = 1) 

    gsRFC = gsRFC.fit(X_train, y_train)

    modelRFC = gsRFC.best_estimator_
    # ########################
    result = permutation_importance(
        gsRFC, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
    )
    ax.set_title("RFC: Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()
    # #######################

    y_pred_RFC = modelRFC.predict(X_valid)
    print('RFC: ', X_train.columns)
    print()
    print('RandomForestClassifier (RFC): classification_report for validation test')
    print()
    print(classification_report(y_valid,y_pred_RFC))

    accuracy_RFC = cross_val_score(estimator=modelRFC, X=X_train, y=y_train, cv=K_fold, scoring='accuracy')
    f1_RFC = cross_val_score(estimator=modelRFC, X=X_train, y=y_train, cv=K_fold, scoring='f1')
    ROCAUC_RFC = cross_val_score(estimator=modelRFC, X=X_train, y=y_train, cv=K_fold, scoring='roc_auc')
    print(f'RFC_validation test: accuracy - {accuracy_RFC.mean()}, f1 - {f1_RFC.mean()}, roc_auc - {ROCAUC_RFC.mean()}')

#     results_dict = {'type':['RFC'],
#                     'len': [len(X_train.columns)],
#                     'features':[list(X_train.columns)],
#                     'model':[modelRFC], 
#                     'accuracy':[accuracy_RFC.mean()],
#                     'f1':[f1_RFC.mean()],
#                     'ROC-AUC':[ROCAUC_RFC.mean()],
#                     'kaggle_score':[''],
#                    }


#     result = pd.DataFrame.from_dict(results_dict)
#     models_result = models_result.append(result)
    print('#########################################################')

    # ######### DecissionTreeClassifier ################################

    pipe_DTC = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=42))])

    parameter_grid = {'classifier__max_depth':[1,2],
                      'classifier__ccp_alpha':[0.001, 0.1, 100]}

    gsDTC = GridSearchCV(estimator=pipe_DTC, param_grid = parameter_grid, cv=K_fold,scoring="accuracy", n_jobs= -1, verbose = 1) 

    gsDTC = gsDTC.fit(X_train, y_train)

    modelDTC = gsDTC.best_estimator_

    # #######################
    result = permutation_importance(
        gsDTC, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
    )
    ax.set_title("DTC: Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()
    # #######################

    y_pred_DTC = modelDTC.predict(X_valid)
    print('DTC: ', X_train.columns)
    print()
    print('DecisionTreeClassifier (DTC): classification_report for validation test')
    print()
    print(classification_report(y_valid,y_pred_DTC))

    accuracy_DTC = cross_val_score(estimator=modelDTC, X=X_train, y=y_train, cv=K_fold, scoring='accuracy')
    f1_DTC = cross_val_score(estimator=modelDTC, X=X_train, y=y_train, cv=K_fold, scoring='f1')
    ROCAUC_DTC = cross_val_score(estimator=modelDTC, X=X_train, y=y_train, cv=K_fold, scoring='roc_auc')
    print(f'DTC_validation test: accuracy - {accuracy_DTC.mean()}, f1 - {f1_DTC.mean()}, roc_auc - {ROCAUC_DTC.mean()}')

#     results_dict = {'type':['DTC'],
#                     'len': [len(X_train.columns)],
#                     'features':[list(X_train.columns)],
#                     'model':[modelDTC], 
#                     'accuracy':[accuracy_DTC.mean()],
#                     'f1':[f1_DTC.mean()],
#                     'ROC-AUC':[ROCAUC_DTC.mean()],
#                     'kaggle_score':[''],
#                    }


#     result = pd.DataFrame.from_dict(results_dict)
#     models_result = models_result.append(result)
    print('#########################################################')

    # ######### GradientBoostingClassifier ################################

    pipe_GBC = Pipeline(
        steps=[("preprocessor", preprocessor),("classifier", GradientBoostingClassifier(random_state=42))] 
    )

    # GBC.get_params().keys()
    gb_param_grid = {
                   'classifier__criterion': ['friedman_mse', 'squared_error', 'mse'],
                  'classifier__loss' : ["deviance"],
                  'classifier__n_estimators' : [sp_randint.rvs(100,500,100)],
                  'classifier__learning_rate': [uniform.rvs(0.001, 0.1)],
                  'classifier__max_depth': [sp_randint.rvs(4,16, 4)],
                  'classifier__min_samples_leaf': [sp_randint.rvs(100,250,50)],
                  'classifier__max_features': [0.3, 0.1]
                  }

    gsGBC = GridSearchCV(pipe_GBC, param_grid = gb_param_grid, cv=K_fold, 
                         scoring="accuracy", n_jobs= 4, verbose = 1)

    gsGBC.fit(X_train,y_train)

    modelGBC = gsGBC.best_estimator_
    # #######################
    result = permutation_importance(
        gsGBC, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
    )
    ax.set_title("GBC: Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()
    # #######################

    y_pred_GBC = modelGBC.predict(X_valid)
    print('GDC: ', X_train.columns)
    print()
    print('GradientBoostingClassifier (GBC): classification_report for validation test')
    print()
    print(classification_report(y_valid,y_pred_GBC, labels=np.unique(y_pred_GBC)))
 
    accuracy_GBC = cross_val_score(estimator=modelGBC, X=X_train, y=y_train, cv=K_fold, scoring='accuracy')
    f1_GBC = cross_val_score(estimator=modelGBC, X=X_train, y=y_train, cv=K_fold, scoring='f1')
    ROCAUC_GBC = cross_val_score(estimator=modelGBC, X=X_train, y=y_train, cv=K_fold, scoring='roc_auc')
    print(f'GBC_validation test: accuracy - {accuracy_GBC.mean()}, f1 - {f1_GBC.mean()}, roc_auc - {ROCAUC_GBC.mean()}')


#     results_dict = {'type':['GBC'],
#                     'len': [len(X_train.columns)],
#                     'features':[list(X_train.columns)],
#                     'model':[modelGBC], 
#                     'accuracy':[accuracy_GBC.mean()],
#                     'f1':[f1_GBC.mean()],
#                     'ROC-AUC':[ROCAUC_GBC.mean()],
#                     'kaggle_score':[''],
#                    }


#     result = pd.DataFrame.from_dict(results_dict)
#     models_result = models_result.append(result)
    print('#########################################################')

    # ########## VotingClassifier #####################################

    VotingPredictor = VotingClassifier(estimators =
                               [ 
                                ('dtc', modelDTC),
                                ('rfc', modelRFC),
                               ],
                               voting='soft', n_jobs = 4)

    VotingPredictor = VotingPredictor.fit(X_train, y_train)

    accuracy_VOT = cross_val_score(estimator=VotingPredictor, X=X_train, y=y_train, cv=K_fold, scoring='accuracy')
    f1_VOT = cross_val_score(estimator=VotingPredictor, X=X_train, y=y_train, cv=K_fold, scoring='f1')
    ROCAUC_VOT = cross_val_score(estimator=VotingPredictor, X=X_train, y=y_train, cv=K_fold, scoring='roc_auc')

    print('VotingClassifier: ', X_train.columns)
    print(f'VotingClassifier_validation test: accuracy - {accuracy_VOT.mean()}, f1 - {f1_VOT.mean()}, roc_auc - {ROCAUC_VOT.mean()}')

#     results_dict = {'type':['VotingClassifier'],
#                     'len': [len(X_train.columns)],
#                     'features':[list(X_train.columns)],
#                     'model':[VotingPredictor], 
#                     'accuracy':[accuracy_VOT.mean()],
#                     'f1':[f1_VOT.mean()],
#                     'ROC-AUC':[ROCAUC_VOT.mean()],
#                     'kaggle_score':[''],
#                    }

#     result = pd.DataFrame.from_dict(results_dict)
#     models_result = models_result.append(result)
    print('#########################################################')
    print()
    return modelRFC, modelDTC, modelGBC, VotingPredictor
    from itertools import combinations
# запускает ф-ю models в виде 1 или 2 варианта
train = trd.copy()

""""
Вариант1: 
 - составляет все возможные комбинации признаков без повторения, без учета порядка (АВ=ВА), мин кол-во признаков = 3
 - отправляет полученные комбинации в ф-ю models
"""

#subset_feature =['Sex','Pclass','Fare','Fare_category','Embarked','Age_new','Family1','Family2','Age_cat2']
# count = 0
# stop, start = 3, len(subset_feature)+1
# for r in range(stop, start):
#     features_combinations = list(combinations(subset_feature, r))
#     count+=1
#     for features in features_combinations:
#         X = train[list(features)]
#         y = train.Survived
#         print(f'# {count} ################################################')
#         modelRFC,modelDTC,modelGBC,VotingPredictor   = models(X, y)
 
""""
Вариант2: отправляет лист признаков в ф-ю models
"""
subset_feature = ['Sex', 'Pclass', 'Fare']
X = train[subset_feature]
y = train.Survived

modelRFC,modelDTC,modelGBC,VotingPredictor  = models(X, y)

# переустановка индексов
# models_result.reset_index(inplace=True, drop=True)
models_result.to_csv('kaggle_titanic_all_features_permutation.csv')
