# Author: Patrick Rocha
# Date: April 8th, 2024
# Student Number: 251168152
# Student ID: procha2

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    # Getting the data and making minor adjustments
    emissionsData = pd.read_csv('my2015-2019-fuel-consumption-ratings.csv')
    wineData = pd.read_csv('winequalityN.csv')
    incomeData = pd.read_csv('11100239.csv')
    housingData = pd.read_csv('Housing_Price_Data.csv')
    emissionsData.columns = [c.replace(' ', '_') for c in emissionsData.columns]
    wineData.columns = [c.replace(' ', '_') for c in wineData.columns]
    incomeData.columns = [c.replace(' ', '_') for c in incomeData.columns]
    emissionsData.rename(columns={'CO2_emissions_(g/km)': 'CO2_emissions'}, inplace=True)

    # Testing the removal of an Emissions column
    emissionsData.drop(columns=['Model', 'Make', 'Transmission'], inplace=True)

    # Removing rows with missing info in the target column
    incomeData = incomeData.dropna(subset=['VALUE'])

    # Getting basic information about the data
    print('\nEmissions Dataset Information')
    print(f'Shape: {emissionsData.shape}')
    print(emissionsData.describe())
    print('\n------------------------------------------------')
    print('\nWine Dataset Information')
    print(f'Shape: {wineData.shape}')
    print(wineData.describe())
    print('\n------------------------------------------------')
    print('\nIncome Dataset Information')
    print(f'Shape: {incomeData.shape}')
    print(incomeData.describe())
    print('\n------------------------------------------------')
    print('\nHousing Dataset Information')
    print(f'Shape: {housingData.shape}')
    print(housingData.describe())

    # Splitting the data
    emissions_y = emissionsData.CO2_emissions
    wine_y = wineData.quality
    income_y = incomeData.VALUE
    housing_y = housingData.price
    emissions_X = emissionsData.drop(['CO2_emissions'], axis=1)
    wine_X = wineData.drop(['quality'], axis=1)
    incomeColumnsToExclude = ['DGUID', 'UOM', 'SCALAR_FACTOR', 'VECTOR', 'COORDINATE', 'VALUE', 'STATUS', 'SYMBOL', 'TERMINATED', 'DECIMALS']
    income_X = incomeData.drop(columns=incomeColumnsToExclude, axis=1)
    housing_X = housingData.drop(['price'], axis=1)

    # Getting a list of the categorical columns and numerical columns
    emissionsCategoricalColumns = emissions_X.select_dtypes(include='object').columns.tolist()
    emissionsNumericalColumns = emissions_X.select_dtypes(include='number').columns.tolist()
    wineCategoricalColumns = wine_X.select_dtypes(include='object').columns.tolist()
    wineNumericalColumns = wine_X.select_dtypes(include='number').columns.tolist()
    incomeCategoricalColumns = income_X.select_dtypes(include='object').columns.tolist()
    incomeNumericalColumns = income_X.select_dtypes(include='number').columns.tolist()
    housingCategoricalColumns = housing_X.select_dtypes(include='object').columns.tolist()
    housingNumericalColumns = housing_X.select_dtypes(include='number').columns.tolist()
    
    print(f'Num categorical: {len(incomeCategoricalColumns)}')
    print(f'Num numerical: {len(incomeNumericalColumns)}')

    #emissions_X = emissionsData.select_dtypes(include='number').drop(['CO2_emissions'], axis=1)
    emissions_X_train, emissions_X_valid, emissions_y_train, emissions_y_valid = train_test_split(emissions_X, emissions_y, train_size=0.8, test_size=0.2, random_state=0)
    wine_X_train, wine_X_valid, wine_y_train, wine_y_valid = train_test_split(wine_X, wine_y, train_size=0.8, test_size=0.2, random_state=0)
    income_X_train, income_X_valid, income_y_train, income_y_valid = train_test_split(income_X, income_y, train_size=0.8, test_size=0.2, random_state=0)
    housing_X_train, housing_X_valid, housing_y_train, housing_y_valid = train_test_split(housing_X, housing_y, train_size=0.8, test_size=0.2, random_state=0)
 
    # Creating copies of the original X
    emissions_X_train_plus = emissions_X_train.copy()
    emissions_X_valid_plus = emissions_X_valid.copy()
    wine_X_train_plus = wine_X_train.copy()
    wine_X_valid_plus = wine_X_valid.copy()
    income_X_train_plus = income_X_train.copy()
    income_X_valid_plus = income_X_valid.copy()
    housing_X_train_plus = housing_X_train.copy()
    housing_X_valid_plus = housing_X_valid.copy()

    emissionMissingColumns = emissionsData.columns[emissionsData.isna().any()].tolist()
    wineMissingColumns = wineData.columns[wineData.isnull().any()].tolist()
    incomeMissingColumns = incomeData.columns[incomeData.isnull().any()].tolist()

    for column in emissionMissingColumns:
        if column in emissionsNumericalColumns:
            # Making new columns indicating what will be imputed
            emissions_X_train_plus[column + '_was_missing'] = emissions_X_train_plus[column].isna().astype(int)
            emissions_X_valid_plus[column + '_was_missing'] = emissions_X_valid_plus[column].isna().astype(int)

            # Impute missing values in the missing columns
            imputer = SimpleImputer(strategy='mean')
            imputed_emissions_X_train_column = imputer.fit_transform(emissions_X_train_plus[[column]])
            imputed_emissions_X_valid_column = imputer.transform(emissions_X_valid_plus[[column]])

            # Imputation removed column names; put them back
            emissions_X_train_plus[column] = imputed_emissions_X_train_column
            emissions_X_valid_plus[column] = imputed_emissions_X_valid_column

    for column in wineMissingColumns:
        # Making new columns indicating what will be imputed
        wine_X_train_plus[column + '_was_missing'] = wine_X_train_plus[column].isnull().astype(int)
        wine_X_valid_plus[column + '_was_missing'] = wine_X_valid_plus[column].isnull().astype(int)

        # Impute missing values in the missing columns
        imputer = SimpleImputer(strategy='mean')
        imputed_wine_X_train_column = imputer.fit_transform(wine_X_train_plus[[column]])
        imputed_wine_X_valid_column = imputer.transform(wine_X_valid_plus[[column]])

        # Imputation removed column names; put them back
        wine_X_train_plus[column] = imputed_wine_X_train_column
        wine_X_valid_plus[column] = imputed_wine_X_valid_column

    for column in incomeMissingColumns:
        if column not in incomeColumnsToExclude:
            # Making new columns indicating what will be imputed
            income_X_train_plus[column + '_was_missing'] = income_X_train_plus[column].isnull().astype(int)
            income_X_valid_plus[column + '_was_missing'] = income_X_valid_plus[column].isnull().astype(int)

            # Impute missing values in the missing columns
            imputer = SimpleImputer(strategy='mean')
            imputed_income_X_train_column = imputer.fit_transform(income_X_train_plus[[column]])
            imputed_income_X_valid_column = imputer.transform(income_X_valid_plus[[column]])

            # Imputation removed column names; put them back
            income_X_train_plus[column] = imputed_income_X_train_column
            income_X_valid_plus[column] = imputed_income_X_valid_column

    # Creating the one hot encoder
    oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Creating empty dataframes for later
    emissions_X_train_OH_columns = pd.DataFrame()
    emissions_X_valid_OH_columns = pd.DataFrame()
    wine_X_train_OH_columns = pd.DataFrame()
    wine_X_valid_OH_columns = pd.DataFrame()
    income_X_train_OH_columns = pd.DataFrame()
    income_X_valid_OH_columns = pd.DataFrame()
    housing_X_train_OH_columns = pd.DataFrame()
    housing_X_valid_OH_columns = pd.DataFrame()

    for column in emissionsCategoricalColumns:
        emissions_X_train_OH_column = pd.DataFrame(oneHotEncoder.fit_transform(emissions_X_train_plus[[column]]))
        emissions_X_valid_OH_column = pd.DataFrame(oneHotEncoder.transform(emissions_X_valid_plus[[column]]))
        
        emissions_encoded_column_names = oneHotEncoder.get_feature_names_out([column])
        
        emissions_X_train_OH_column.columns = emissions_encoded_column_names
        emissions_X_valid_OH_column.columns = emissions_encoded_column_names  

        emissions_X_train_OH_columns = pd.concat([emissions_X_train_OH_columns, emissions_X_train_OH_column], axis=1)
        emissions_X_valid_OH_columns = pd.concat([emissions_X_valid_OH_columns, emissions_X_valid_OH_column], axis=1)

    for column in wineCategoricalColumns:
        wine_X_train_OH_column = pd.DataFrame(oneHotEncoder.fit_transform(wine_X_train_plus[[column]]))
        wine_X_valid_OH_column = pd.DataFrame(oneHotEncoder.transform(wine_X_valid_plus[[column]]))
        
        wine_encoded_column_names = oneHotEncoder.get_feature_names_out([column])
        
        wine_X_train_OH_column.columns = wine_encoded_column_names
        wine_X_valid_OH_column.columns = wine_encoded_column_names  

        wine_X_train_OH_columns = pd.concat([wine_X_train_OH_columns, wine_X_train_OH_column], axis=1)
        wine_X_valid_OH_columns = pd.concat([wine_X_valid_OH_columns, wine_X_valid_OH_column], axis=1)

    for column in incomeCategoricalColumns:
        income_X_train_OH_column = pd.DataFrame(oneHotEncoder.fit_transform(income_X_train_plus[[column]]))
        income_X_valid_OH_column = pd.DataFrame(oneHotEncoder.transform(income_X_valid_plus[[column]]))
        
        income_encoded_column_names = oneHotEncoder.get_feature_names_out([column])
        
        income_X_train_OH_column.columns = income_encoded_column_names
        income_X_valid_OH_column.columns = income_encoded_column_names  

        income_X_train_OH_columns = pd.concat([income_X_train_OH_columns, income_X_train_OH_column], axis=1)
        income_X_valid_OH_columns = pd.concat([income_X_valid_OH_columns, income_X_valid_OH_column], axis=1)

    for column in housingCategoricalColumns:
        housing_X_train_OH_column = pd.DataFrame(oneHotEncoder.fit_transform(housing_X_train_plus[[column]]))
        housing_X_valid_OH_column = pd.DataFrame(oneHotEncoder.transform(housing_X_valid_plus[[column]]))
        
        housing_encoded_column_names = oneHotEncoder.get_feature_names_out([column])
        
        housing_X_train_OH_column.columns = housing_encoded_column_names
        housing_X_valid_OH_column.columns = housing_encoded_column_names  

        housing_X_train_OH_columns = pd.concat([housing_X_train_OH_columns, housing_X_train_OH_column], axis=1)
        housing_X_valid_OH_columns = pd.concat([housing_X_valid_OH_columns, housing_X_valid_OH_column], axis=1)

    # Bringing back the index
    emissions_X_train_OH_columns.index = emissions_X_train_plus.index
    emissions_X_valid_OH_columns.index = emissions_X_valid_plus.index
    wine_X_train_OH_columns.index = wine_X_train_plus.index
    wine_X_valid_OH_columns.index = wine_X_valid_plus.index
    income_X_train_OH_columns.index = income_X_train_plus.index
    income_X_valid_OH_columns.index = income_X_valid_plus.index
    housing_X_train_OH_columns.index = housing_X_train_plus.index
    housing_X_valid_OH_columns.index = housing_X_valid_plus.index

    # Drop categorical columns
    emissions_X_train_temp = emissions_X_train_plus.drop(emissionsCategoricalColumns, axis=1)
    emissions_X_valid_temp = emissions_X_valid_plus.drop(emissionsCategoricalColumns, axis=1)
    wine_X_train_temp = wine_X_train_plus.drop(wineCategoricalColumns, axis=1)
    wine_X_valid_temp = wine_X_valid_plus.drop(wineCategoricalColumns, axis=1)
    income_X_train_temp = income_X_train_plus.drop(incomeCategoricalColumns, axis=1)
    income_X_valid_temp = income_X_valid_plus.drop(incomeCategoricalColumns, axis=1)
    housing_X_train_temp = housing_X_train_plus.drop(housingCategoricalColumns, axis=1)
    housing_X_valid_temp = housing_X_valid_plus.drop(housingCategoricalColumns, axis=1)
    
    # Concatenating the numerical columns with the new one-hot-encoded columns
    emissions_X_train_OH = pd.concat([emissions_X_train_temp, emissions_X_train_OH_columns], axis=1)
    emissions_X_valid_OH = pd.concat([emissions_X_valid_temp, emissions_X_valid_OH_columns], axis=1)
    wine_X_train_OH = pd.concat([wine_X_train_temp, wine_X_train_OH_columns], axis=1)
    wine_X_valid_OH = pd.concat([wine_X_valid_temp, wine_X_valid_OH_columns], axis=1)
    income_X_train_OH = pd.concat([income_X_train_temp, income_X_train_OH_columns], axis=1)
    income_X_valid_OH = pd.concat([income_X_valid_temp, income_X_valid_OH_columns], axis=1)
    housing_X_train_OH = pd.concat([housing_X_train_temp, housing_X_train_OH_columns], axis=1)
    housing_X_valid_OH = pd.concat([housing_X_valid_temp, housing_X_valid_OH_columns], axis=1)

    emissions_X_train_OH.columns = emissions_X_train_OH.columns.astype(str)
    emissions_X_valid_OH.columns = emissions_X_valid_OH.columns.astype(str)
    wine_X_train_OH.columns = wine_X_train_OH.columns.astype(str)
    wine_X_valid_OH.columns = wine_X_valid_OH.columns.astype(str)
    income_X_train_OH.columns = income_X_train_OH.columns.astype(str)
    income_X_valid_OH.columns = income_X_valid_OH.columns.astype(str)
    housing_X_train_OH.columns = housing_X_train_OH.columns.astype(str)
    housing_X_valid_OH.columns = housing_X_valid_OH.columns.astype(str)

    # Drop categorical columns permanetly to test without OH encoding
    emissions_X_train_plus = emissions_X_train_plus.select_dtypes(exclude='object')
    emissions_X_valid_plus = emissions_X_valid_plus.select_dtypes(exclude='object')
    wine_X_train_plus = wine_X_train_plus.select_dtypes(exclude='object')
    wine_X_valid_plus = wine_X_valid_plus.select_dtypes(exclude='object')
    income_X_train_plus = income_X_train_plus.select_dtypes(exclude='object')
    income_X_valid_plus = income_X_valid_plus.select_dtypes(exclude='object')
    housing_X_train_plus = housing_X_train_plus.select_dtypes(exclude='object')
    housing_X_valid_plus = housing_X_valid_plus.select_dtypes(exclude='object')

    """
    datasets = [
        (emissions_X_train_OH, emissions_y_train),
        (wine_X_train_OH, wine_y_train),
        (housing_X_train_OH, housing_y_train)
        (income_X_train_OH, income_y_train)
    ]

    bestParameters = {}

    for i, (X_train, y_train) in enumerate(datasets):
        # Creating parameter grids for the model parameters
        randomForestParamGrid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        kNeighborParamGrid = {
            'n_neighbors': [3, 5, 7, 9] 
        }
        lassoParamGrid = {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }
        ridgeParamGrid = {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }

        # Creating the models
        randomForestModel = RandomForestRegressor()
        linearRegressionModel = LinearRegression()
        decisionTreeModel = DecisionTreeRegressor()
        kNeighborModel = KNeighborsRegressor()
        lassoModel = Lasso()
        ridgeModel = Ridge()

        # Grouping the models and param grids
        models = [
            (randomForestModel, randomForestParamGrid),
            (linearRegressionModel, {}),
            (decisionTreeModel, {}),
            (kNeighborModel, kNeighborParamGrid),
            (lassoModel, lassoParamGrid),
            (ridgeModel, ridgeParamGrid)
        ]

        datasetParameters = {}
        for model, paramGrid in models:
            gridSearch = GridSearchCV(estimator=model, param_grid=paramGrid, scoring='neg_mean_squared_error', cv=5)
            gridSearch.fit(X_train, y_train)
            datasetParameters[type(model).__name__] = gridSearch.best_params_

        bestParameters[f'{i + 1}'] = datasetParameters

    for dataset_name, model_params in bestParameters.items():
        print(f"Dataset: {dataset_name}")
        # Iterate over each model and its parameters
        for model_name, params in model_params.items():
            print(f"  Model: {model_name}")
            # Print each parameter and its value
            for param_name, param_value in params.items():
                print(f"    {param_name}: {param_value}")
    """

    # Re-creating the models with the optimal parameters
    randomForestModel = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=150, random_state=1)
    kNeighborModel = KNeighborsRegressor(n_neighbors=3)
    linearRegressionModel = LinearRegression()
    decisionTreeModel = DecisionTreeRegressor()
    lassoModel = Lasso(alpha=0.01, random_state=1)
    ridgeModel = Ridge(alpha=1.0, random_state=1)

    print('\n\n---------------------------------------------------------------------------------------------------------------------------------')
    print("\nEMISSIONS SCORES")
    #print(f'\nNumber of unique columns in Make: {emissionsData['Make'].nunique()}')
    #print(f'Number of unique columns in Model: {emissionsData['Model'].nunique()}')
    #print(f'Number of unique columns in Vehcile Class: {emissionsData['Vehicle_class'].nunique()}')
    #print(f'Number of unique columns in Transmission: {emissionsData['Transmission'].nunique()}')
    #print(f'Number of unique columns in Fuel Type: {emissionsData['Fuel_type'].nunique()}')
    emissionsRandomForestScores = -1 * cross_val_score(randomForestModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest MAE scores: {emissionsRandomForestScores}')
    print(f'Mean: {emissionsRandomForestScores.mean()}')
    print(f'Standard Deviation: {emissionsRandomForestScores.std()}')
    emissionsRandomForestScores = -1 * cross_val_score(randomForestModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH MAE scores: {emissionsRandomForestScores}')
    print(f'Mean: {emissionsRandomForestScores.mean()}')
    print(f'Standard Deviation: {emissionsRandomForestScores.std()}')
    emissionsRandomForestScores = cross_val_score(randomForestModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest R2 scores: {emissionsRandomForestScores}')
    print(f'Mean: {emissionsRandomForestScores.mean()}')
    print(f'Standard Deviation: {emissionsRandomForestScores.std()}')
    emissionsRandomForestScores = cross_val_score(randomForestModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH R2 scores: {emissionsRandomForestScores}')
    print(f'Mean: {emissionsRandomForestScores.mean()}')
    print(f'Standard Deviation: {emissionsRandomForestScores.std()}')

    emissionsLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression MAE scores: {emissionsLinearRegressionScores}')
    print(f'Mean: {emissionsLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {emissionsLinearRegressionScores.std()}')
    emissionsLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH MAE scores: {emissionsLinearRegressionScores}')
    print(f'Mean: {emissionsLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {emissionsLinearRegressionScores.std()}')
    emissionsLinearRegressionScores = cross_val_score(linearRegressionModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression R2 scores: {emissionsLinearRegressionScores}')
    print(f'Mean: {emissionsLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {emissionsLinearRegressionScores.std()}')
    emissionsLinearRegressionScores = cross_val_score(linearRegressionModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH R2 scores: {emissionsLinearRegressionScores}')
    print(f'Mean: {emissionsLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {emissionsLinearRegressionScores.std()}')

    emissionsDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree MAE scores: {emissionsDecisionTreeScores}')
    print(f'Mean: {emissionsDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {emissionsDecisionTreeScores.std()}')
    emissionsDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH MAE scores: {emissionsDecisionTreeScores}')
    print(f'Mean: {emissionsDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {emissionsDecisionTreeScores.std()}')
    emissionsDecisionTreeScores = cross_val_score(decisionTreeModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree R2 scores: {emissionsDecisionTreeScores}')
    print(f'Mean: {emissionsDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {emissionsDecisionTreeScores.std()}')
    emissionsDecisionTreeScores = cross_val_score(decisionTreeModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH R2 scores: {emissionsDecisionTreeScores}')
    print(f'Mean: {emissionsDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {emissionsDecisionTreeScores.std()}')

    emissionskNeighbourScores = -1 * cross_val_score(kNeighborModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour MAE scores: {emissionskNeighbourScores}')
    print(f'Mean: {emissionskNeighbourScores.mean()}')
    print(f'Standard Deviation: {emissionskNeighbourScores.std()}')
    emissionskNeighbourScores = -1 * cross_val_score(kNeighborModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH MAE scores: {emissionskNeighbourScores}')
    print(f'Mean: {emissionskNeighbourScores.mean()}')
    print(f'Standard Deviation: {emissionskNeighbourScores.std()}')
    emissionskNeighbourScores = cross_val_score(kNeighborModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour R2 scores: {emissionskNeighbourScores}')
    print(f'Mean: {emissionskNeighbourScores.mean()}')
    print(f'Standard Deviation: {emissionskNeighbourScores.std()}')
    emissionskNeighbourScores = cross_val_score(kNeighborModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH R2 scores: {emissionskNeighbourScores}')
    print(f'Mean: {emissionskNeighbourScores.mean()}')
    print(f'Standard Deviation: {emissionskNeighbourScores.std()}')

    emissionsLassoScores = -1 * cross_val_score(lassoModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso MAE scores: {emissionsLassoScores}')
    print(f'Mean: {emissionsLassoScores.mean()}')
    print(f'Standard Deviation: {emissionsLassoScores.std()}')
    emissionsLassoScores = -1 * cross_val_score(lassoModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso OH MAE scores: {emissionsLassoScores}')
    print(f'Mean: {emissionsLassoScores.mean()}')
    print(f'Standard Deviation: {emissionsLassoScores.std()}')
    emissionsLassoScores = cross_val_score(lassoModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso R2 scores: {emissionsLassoScores}')
    print(f'Mean: {emissionsLassoScores.mean()}')
    print(f'Standard Deviation: {emissionsLassoScores.std()}')
    emissionsLassoScores = cross_val_score(lassoModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso OH R2 scores: {emissionsLassoScores}')
    print(f'Mean: {emissionsLassoScores.mean()}')
    print(f'Standard Deviation: {emissionsLassoScores.std()}')

    emissionsRidgeScores = -1 * cross_val_score(ridgeModel, emissions_X_train_plus, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge MAE scores: {emissionsRidgeScores}')
    print(f'Mean: {emissionsRidgeScores.mean()}')
    print(f'Standard Deviation: {emissionsRidgeScores.std()}')
    emissionsRidgeScores = -1 * cross_val_score(ridgeModel, emissions_X_train_OH, emissions_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge OH MAE scores: {emissionsRidgeScores}')
    print(f'Mean: {emissionsRidgeScores.mean()}')
    print(f'Standard Deviation: {emissionsRidgeScores.std()}')
    emissionsRidgeScores = cross_val_score(ridgeModel, emissions_X_train_plus, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge R2 scores: {emissionsRidgeScores}')
    print(f'Mean: {emissionsRidgeScores.mean()}')
    print(f'Standard Deviation: {emissionsRidgeScores.std()}')
    emissionsRidgeScores = cross_val_score(ridgeModel, emissions_X_train_OH, emissions_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge OH R2 scores: {emissionsRidgeScores}')
    print(f'Mean: {emissionsRidgeScores.mean()}')
    print(f'Standard Deviation: {emissionsRidgeScores.std()}')

    #-----------------------------------------------------------------------------------------------------------------
    # Re-creating the models with the optimal parameters
    randomForestModel = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=150, random_state=1)
    kNeighborModel = KNeighborsRegressor(n_neighbors=9)
    lassoModel = Lasso(alpha=0.001, random_state=1)
    ridgeModel = Ridge(alpha=0.001, random_state=1)

    print('---------------------------------------------------------------------------------------------------------------------------------')
    print("\nWINE SCORES")
    wineRandomForestScores = -1 * cross_val_score(randomForestModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest MAE scores: {wineRandomForestScores}')
    print(f'Mean: {wineRandomForestScores.mean()}')
    print(f'Standard Deviation: {wineRandomForestScores.std()}')
    wineRandomForestScores = -1 * cross_val_score(randomForestModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH MAE scores: {wineRandomForestScores}')
    print(f'Mean: {wineRandomForestScores.mean()}')
    print(f'Standard Deviation: {wineRandomForestScores.std()}')
    wineRandomForestScores = cross_val_score(randomForestModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest R2 scores: {wineRandomForestScores}')
    print(f'Mean: {wineRandomForestScores.mean()}')
    print(f'Standard Deviation: {wineRandomForestScores.std()}')
    wineRandomForestScores = cross_val_score(randomForestModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH R2 scores: {wineRandomForestScores}')
    print(f'Mean: {wineRandomForestScores.mean()}')
    print(f'Standard Deviation: {wineRandomForestScores.std()}')

    wineLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression MAE scores: {wineLinearRegressionScores}')
    print(f'Mean: {wineLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {wineLinearRegressionScores.std()}')
    wineLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH MAE scores: {wineLinearRegressionScores}')
    print(f'Mean: {wineLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {wineLinearRegressionScores.std()}')
    wineLinearRegressionScores = cross_val_score(linearRegressionModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression R2 scores: {wineLinearRegressionScores}')
    print(f'Mean: {wineLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {wineLinearRegressionScores.std()}')
    wineLinearRegressionScores = cross_val_score(linearRegressionModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH R2 scores: {wineLinearRegressionScores}')
    print(f'Mean: {wineLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {wineLinearRegressionScores.std()}')

    wineDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree MAE scores: {wineDecisionTreeScores}')
    print(f'Mean: {wineDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {wineDecisionTreeScores.std()}')
    wineDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH MAE scores: {wineDecisionTreeScores}')
    print(f'Mean: {wineDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {wineDecisionTreeScores.std()}')
    wineDecisionTreeScores = cross_val_score(decisionTreeModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree R2 scores: {wineDecisionTreeScores}')
    print(f'Mean: {wineDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {wineDecisionTreeScores.std()}')
    wineDecisionTreeScores = cross_val_score(decisionTreeModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH R2 scores: {wineDecisionTreeScores}')
    print(f'Mean: {wineDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {wineDecisionTreeScores.std()}')

    winekNeighbourScores = -1 * cross_val_score(kNeighborModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour MAE scores: {winekNeighbourScores}')
    print(f'Mean: {winekNeighbourScores.mean()}')
    print(f'Standard Deviation: {winekNeighbourScores.std()}')
    winekNeighbourScores = -1 * cross_val_score(kNeighborModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH MAE scores: {winekNeighbourScores}')
    print(f'Mean: {winekNeighbourScores.mean()}')
    print(f'Standard Deviation: {winekNeighbourScores.std()}')
    winekNeighbourScores = cross_val_score(kNeighborModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour R2 scores: {winekNeighbourScores}')
    print(f'Mean: {winekNeighbourScores.mean()}')
    print(f'Standard Deviation: {winekNeighbourScores.std()}')
    winekNeighbourScores = cross_val_score(kNeighborModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH R2 scores: {winekNeighbourScores}')
    print(f'Mean: {winekNeighbourScores.mean()}')
    print(f'Standard Deviation: {winekNeighbourScores.std()}')

    wineLassoScores = -1 * cross_val_score(lassoModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso MAE scores: {wineLassoScores}')
    print(f'Mean: {wineLassoScores.mean()}')
    print(f'Standard Deviation: {wineLassoScores.std()}')
    wineLassoScores = -1 * cross_val_score(lassoModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso OH MAE scores: {wineLassoScores}')
    print(f'Mean: {wineLassoScores.mean()}')
    print(f'Standard Deviation: {wineLassoScores.std()}')
    wineLassoScores = cross_val_score(lassoModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso R2 scores: {wineLassoScores}')
    print(f'Mean: {wineLassoScores.mean()}')
    print(f'Standard Deviation: {wineLassoScores.std()}')
    wineLassoScores = cross_val_score(lassoModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso OH R2 scores: {wineLassoScores}')
    print(f'Mean: {wineLassoScores.mean()}')
    print(f'Standard Deviation: {wineLassoScores.std()}')

    wineRidgeScores = -1 * cross_val_score(ridgeModel, wine_X_train_plus, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge MAE scores: {wineRidgeScores}')
    print(f'Mean: {wineRidgeScores.mean()}')
    print(f'Standard Deviation: {wineRidgeScores.std()}')
    wineRidgeScores = -1 * cross_val_score(ridgeModel, wine_X_train_OH, wine_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge OH MAE scores: {wineRidgeScores}')
    print(f'Mean: {wineRidgeScores.mean()}')
    print(f'Standard Deviation: {wineRidgeScores.std()}')
    wineRidgeScores = cross_val_score(ridgeModel, wine_X_train_plus, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge R2 scores: {wineRidgeScores}')
    print(f'Mean: {wineRidgeScores.mean()}')
    print(f'Standard Deviation: {wineRidgeScores.std()}')
    wineRidgeScores = cross_val_score(ridgeModel, wine_X_train_OH, wine_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge OH R2 scores: {wineRidgeScores}')
    print(f'Mean: {wineRidgeScores.mean()}')
    print(f'Standard Deviation: {wineRidgeScores.std()}')

    #-----------------------------------------------------------------------------------------------------------------
    randomForestModel = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=150, random_state=1)
    kNeighborModel = KNeighborsRegressor(n_neighbors=9)
    lassoModel = Lasso(alpha=1.0, random_state=1)
    ridgeModel = Ridge(alpha=1.0, random_state=1)

    print('---------------------------------------------------------------------------------------------------------------------------------')
    print("\nINCOME SCORES:")
    incomeRandomForestScores = -1 * cross_val_score(randomForestModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest MAE scores: {incomeRandomForestScores}')
    print(f'Mean: {incomeRandomForestScores.mean()}')
    print(f'Standard Deviation: {incomeRandomForestScores.std()}')
    incomeRandomForestScores = -1 * cross_val_score(randomForestModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH MAE scores: {incomeRandomForestScores}')
    print(f'Mean: {incomeRandomForestScores.mean()}')
    print(f'Standard Deviation: {incomeRandomForestScores.std()}')
    incomeRandomForestScores = cross_val_score(randomForestModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest R2 scores: {incomeRandomForestScores}')
    print(f'Mean: {incomeRandomForestScores.mean()}')
    print(f'Standard Deviation: {incomeRandomForestScores.std()}')
    incomeRandomForestScores = cross_val_score(randomForestModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH R2 scores: {incomeRandomForestScores}')
    print(f'Mean: {incomeRandomForestScores.mean()}')
    print(f'Standard Deviation: {incomeRandomForestScores.std()}')

    incomeLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression MAE scores: {incomeLinearRegressionScores}')
    print(f'Mean: {incomeLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {incomeLinearRegressionScores.std()}')
    incomeLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH MAE scores: {incomeLinearRegressionScores}')
    print(f'Mean: {incomeLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {incomeLinearRegressionScores.std()}')
    incomeLinearRegressionScores = cross_val_score(linearRegressionModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression R2 scores: {incomeLinearRegressionScores}')
    print(f'Mean: {incomeLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {incomeLinearRegressionScores.std()}')
    incomeLinearRegressionScores = cross_val_score(linearRegressionModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH R2 scores: {incomeLinearRegressionScores}')
    print(f'Mean: {incomeLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {incomeLinearRegressionScores.std()}')

    incomeDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree MAE scores: {incomeDecisionTreeScores}')
    print(f'Mean: {incomeDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {incomeDecisionTreeScores.std()}')
    incomeDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH MAE scores: {incomeDecisionTreeScores}')
    print(f'Mean: {incomeDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {incomeDecisionTreeScores.std()}')
    incomeDecisionTreeScores = cross_val_score(decisionTreeModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree R2 scores: {incomeDecisionTreeScores}')
    print(f'Mean: {incomeDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {incomeDecisionTreeScores.std()}')
    incomeDecisionTreeScores = cross_val_score(decisionTreeModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH R2 scores: {incomeDecisionTreeScores}')
    print(f'Mean: {incomeDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {incomeDecisionTreeScores.std()}')

    incomekNeighbourScores = -1 * cross_val_score(kNeighborModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour MAE scores: {incomekNeighbourScores}')
    print(f'Mean: {incomekNeighbourScores.mean()}')
    print(f'Standard Deviation: {incomekNeighbourScores.std()}')
    incomekNeighbourScores = -1 * cross_val_score(kNeighborModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH MAE scores: {incomekNeighbourScores}')
    print(f'Mean: {incomekNeighbourScores.mean()}')
    print(f'Standard Deviation: {incomekNeighbourScores.std()}')
    incomekNeighbourScores = cross_val_score(kNeighborModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour R2 scores: {incomekNeighbourScores}')
    print(f'Mean: {incomekNeighbourScores.mean()}')
    print(f'Standard Deviation: {incomekNeighbourScores.std()}')
    incomekNeighbourScores = cross_val_score(kNeighborModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH R2 scores: {incomekNeighbourScores}')
    print(f'Mean: {incomekNeighbourScores.mean()}')
    print(f'Standard Deviation: {incomekNeighbourScores.std()}')

    incomeLassoScores = -1 * cross_val_score(lassoModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso MAE scores: {incomeLassoScores}')
    print(f'Mean: {incomeLassoScores.mean()}')
    print(f'Standard Deviation: {incomeLassoScores.std()}')
    incomeLassoScores = -1 * cross_val_score(lassoModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso OH MAE scores: {incomeLassoScores}')
    print(f'Mean: {incomeLassoScores.mean()}')
    print(f'Standard Deviation: {incomeLassoScores.std()}')
    incomeLassoScores = cross_val_score(lassoModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso R2 scores: {incomeLassoScores}')
    print(f'Mean: {incomeLassoScores.mean()}')
    print(f'Standard Deviation: {incomeLassoScores.std()}')
    incomeLassoScores = cross_val_score(lassoModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso OH R2 scores: {incomeLassoScores}')
    print(f'Mean: {incomeLassoScores.mean()}')
    print(f'Standard Deviation: {incomeLassoScores.std()}')

    incomeRidgeScores = -1 * cross_val_score(ridgeModel, income_X_train_plus, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge MAE scores: {incomeRidgeScores}')
    print(f'Mean: {incomeRidgeScores.mean()}')
    print(f'Standard Deviation: {incomeRidgeScores.std()}')
    incomeRidgeScores = -1 * cross_val_score(ridgeModel, income_X_train_OH, income_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge OH MAE scores: {incomeRidgeScores}')
    print(f'Mean: {incomeRidgeScores.mean()}')
    print(f'Standard Deviation: {incomeRidgeScores.std()}')
    incomeRidgeScores = cross_val_score(ridgeModel, income_X_train_plus, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge R2 scores: {incomeRidgeScores}')
    print(f'Mean: {incomeRidgeScores.mean()}')
    print(f'Standard Deviation: {incomeRidgeScores.std()}')
    incomeRidgeScores = cross_val_score(ridgeModel, income_X_train_OH, income_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge OH R2 scores: {incomeRidgeScores}')
    print(f'Mean: {incomeRidgeScores.mean()}')
    print(f'Standard Deviation: {incomeRidgeScores.std()}')

    #-----------------------------------------------------------------------------------------------------------------
    # Re-creating the models with the optimal parameters
    randomForestModel = RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=1)
    kNeighborModel = KNeighborsRegressor(n_neighbors=7)
    lassoModel = Lasso(alpha=1.0, random_state=1)
    ridgeModel = Ridge(alpha=1.0, random_state=1)

    print('---------------------------------------------------------------------------------------------------------------------------------')
    print("\nHOUSING SCORES:")

    housingRandomForestScores = -1 * cross_val_score(randomForestModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest MAE scores: {housingRandomForestScores}')
    print(f'Mean: {housingRandomForestScores.mean()}')
    print(f'Standard Deviation: {housingRandomForestScores.std()}')
    housingRandomForestScores = -1 * cross_val_score(randomForestModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH MAE scores: {housingRandomForestScores}')
    print(f'Mean: {housingRandomForestScores.mean()}')
    print(f'Standard Deviation: {housingRandomForestScores.std()}')
    housingRandomForestScores = cross_val_score(randomForestModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest R2 scores: {housingRandomForestScores}')
    print(f'Mean: {housingRandomForestScores.mean()}')
    print(f'Standard Deviation: {housingRandomForestScores.std()}')
    housingRandomForestScores = cross_val_score(randomForestModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRandom Forest OH R2 scores: {housingRandomForestScores}')
    print(f'Mean: {housingRandomForestScores.mean()}')
    print(f'Standard Deviation: {housingRandomForestScores.std()}')

    housingLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression MAE scores: {housingLinearRegressionScores}')
    print(f'Mean: {housingLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {housingLinearRegressionScores.std()}')
    housingLinearRegressionScores = -1 * cross_val_score(linearRegressionModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH MAE scores: {housingLinearRegressionScores}')
    print(f'Mean: {housingLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {housingLinearRegressionScores.std()}')
    housingLinearRegressionScores = cross_val_score(linearRegressionModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression R2 scores: {housingLinearRegressionScores}')
    print(f'Mean: {housingLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {housingLinearRegressionScores.std()}')
    housingLinearRegressionScores = cross_val_score(linearRegressionModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLinear Regression OH R2 scores: {housingLinearRegressionScores}')
    print(f'Mean: {housingLinearRegressionScores.mean()}')
    print(f'Standard Deviation: {housingLinearRegressionScores.std()}')

    housingDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree MAE scores: {housingDecisionTreeScores}')
    print(f'Mean: {housingDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {housingDecisionTreeScores.std()}')
    housingDecisionTreeScores = -1 * cross_val_score(decisionTreeModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH MAE scores: {housingDecisionTreeScores}')
    print(f'Mean: {housingDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {housingDecisionTreeScores.std()}')
    housingDecisionTreeScores = cross_val_score(decisionTreeModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree R2 scores: {housingDecisionTreeScores}')
    print(f'Mean: {housingDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {housingDecisionTreeScores.std()}')
    housingDecisionTreeScores = cross_val_score(decisionTreeModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nDecision Tree OH R2 scores: {housingDecisionTreeScores}')
    print(f'Mean: {housingDecisionTreeScores.mean()}')
    print(f'Standard Deviation: {housingDecisionTreeScores.std()}')

    housingkNeighbourScores = -1 * cross_val_score(kNeighborModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour MAE scores: {housingkNeighbourScores}')
    print(f'Mean: {housingkNeighbourScores.mean()}')
    print(f'Standard Deviation: {housingkNeighbourScores.std()}')
    housingkNeighbourScores = -1 * cross_val_score(kNeighborModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH MAE scores: {housingkNeighbourScores}')
    print(f'Mean: {housingkNeighbourScores.mean()}')
    print(f'Standard Deviation: {housingkNeighbourScores.std()}')
    housingkNeighbourScores = cross_val_score(kNeighborModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour R2 scores: {housingkNeighbourScores}')
    print(f'Mean: {housingkNeighbourScores.mean()}')
    print(f'Standard Deviation: {housingkNeighbourScores.std()}')
    housingkNeighbourScores = cross_val_score(kNeighborModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nK Neighbour OH R2 scores: {housingkNeighbourScores}')
    print(f'Mean: {housingkNeighbourScores.mean()}')
    print(f'Standard Deviation: {housingkNeighbourScores.std()}')

    housingLassoScores = -1 * cross_val_score(lassoModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso MAE scores: {housingLassoScores}')
    print(f'Mean: {housingLassoScores.mean()}')
    print(f'Standard Deviation: {housingLassoScores.std()}')
    housingLassoScores = -1 * cross_val_score(lassoModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nLasso OH MAE scores: {housingLassoScores}')
    print(f'Mean: {housingLassoScores.mean()}')
    print(f'Standard Deviation: {housingLassoScores.std()}')
    housingLassoScores = cross_val_score(lassoModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso R2 scores: {housingLassoScores}')
    print(f'Mean: {housingLassoScores.mean()}')
    print(f'Standard Deviation: {housingLassoScores.std()}')
    housingLassoScores = cross_val_score(lassoModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nLasso OH R2 scores: {housingLassoScores}')
    print(f'Mean: {housingLassoScores.mean()}')
    print(f'Standard Deviation: {housingLassoScores.std()}')

    housingRidgeScores = -1 * cross_val_score(ridgeModel, housing_X_train_plus, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge MAE scores: {housingRidgeScores}')
    print(f'Mean: {housingRidgeScores.mean()}')
    print(f'Standard Deviation: {housingRidgeScores.std()}')
    housingRidgeScores = -1 * cross_val_score(ridgeModel, housing_X_train_OH, housing_y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    print(f'\nRidge OH MAE scores: {housingRidgeScores}')
    print(f'Mean: {housingRidgeScores.mean()}')
    print(f'Standard Deviation: {housingRidgeScores.std()}')
    housingRidgeScores = cross_val_score(ridgeModel, housing_X_train_plus, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge R2 scores: {housingRidgeScores}')
    print(f'Mean: {housingRidgeScores.mean()}')
    print(f'Standard Deviation: {housingRidgeScores.std()}')
    housingRidgeScores = cross_val_score(ridgeModel, housing_X_train_OH, housing_y_train, scoring='r2', cv=5, n_jobs=-1)
    print(f'\nRidge OH R2 scores: {housingRidgeScores}')
    print(f'Mean: {housingRidgeScores.mean()}')
    print(f'Standard Deviation: {housingRidgeScores.std()}')

if __name__ == '__main__':
    main()