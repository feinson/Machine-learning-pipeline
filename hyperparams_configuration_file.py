# Regression model hyperparameters

SGDRegressor_params = [{'loss': ['epsilon_insensitive'],
             'penalty' : ['l1'],
             'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
             'max_iter' : [1500, 2500],
             'learning_rate': ['optimal']}]


LinearRegression_params = [{"fit_intercept": [True, False],
                            "positive": [True, False]}]

DecisionTreeRegressor_params = [{"splitter": ["best","random"],
                        "max_depth": [1, 3, 5, 7],
                        "min_samples_leaf": [5, 6],
                        "min_weight_fraction_leaf": [0.1, 0.2],
                        "max_features": ["sqrt", None],
                        "max_leaf_nodes": [70]}]


RandomForestRegressor_params = [{'bootstrap': [True],
                        'max_depth': [2, 8, 16, 32],
                        'max_features': ['sqrt'],
                        'min_samples_leaf': [4],
                        'min_samples_split': [1, 5, 10],
                        'n_estimators': [100, 200]}]


GradientBoostingRegressor_params = [{'n_estimators': [500, 1000, 2000],
                         'learning_rate': [.001, 0.01],
                         'max_depth': [1, 2, 4],
                         'subsample': [.75, 1],
                         'random_state': [1]}]

#Classification models hyperparameters

LogisticRegression_params = [    
    {'penalty': ['l2'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['newton-cg', 'lbfgs', 'sag'], 'max_iter': [400]},
    {'penalty': ['l1', 'l2'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['saga'], 'max_iter': [400]},
    {'penalty': ['elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['saga'], 'l1_ratio': [0, 0.1, 0.3, 0.6, 0.8, 1], 'max_iter': [400]}]

DecisionTreeClassifier_params = [{'criterion': ['gini', 'entropy'],
                              "splitter": ["best","random"],
                              'max_depth': [2, 8, 16, 32, None],
                              'min_samples_split': range(1, 10),
                              'min_samples_leaf': range(1,5)}]

RandomForestClassifier_params = [{'criterion': ['gini', 'entropy'],
                              'max_depth': [2, 8, 16, None],
                              'min_samples_split': [1, 5, 10],
                              'min_samples_leaf': [1, 5, 10]}]

GradientBoostingClassifier_params = [{'loss': ['log_loss'],
                                  'learning_rate': [.001, 0.01, .1],
                                  'n_estimators': [100, 200],
                                  'subsample': [.75, 1]}]