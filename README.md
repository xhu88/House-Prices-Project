# House Prices Model Card

### Basic Information

* **Group developing model**: Meghana Srinivasaiah `meghanasrinivas@gwu.edu`, Jake Fitzpatrick `jakefitz@gwu.edu`, Jesse Mutamba `jesse.mutamba@gwu.edu` and Xinya Hu `xhu88@gwu.edu`

* **Model date**: December, 2024
* **Model version**: 1.0
* **License**: MIT
* **Model Implementation Code**: [DNSC3288_SemesterProject_Final (1).ipynb](https://github.com/xhu88/House-Prices-Project/blob/53e4ee6645ed0130d59ea918389dedc70dabba15/DNSC3288_SemesterProject_Final%20(1).ipynb)

### Intended Use
* **Primary Intended Uses**: This model is for educational uses only.
* **Primary Intended Users**: Students in DNSC 3288, and students learning about machine learning.
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

### Training Data

* **Source of Training Data**: Kaggle.com, [House Prices Prediction using TFDF](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/input)
* **How training data was divided into test data and validation data**: 40% training, 10% validation, 50% testing
* **Number of rows in training and validation data**:
  * Training rows: 1168
  * Validation rows: 292

* **Data Dictionary**:

|     Name        | Modeling Role  | Measurement Level | Description                                                |
|------------------|----------------|-------------------|------------------------------------------------------------|
| SalePrice        | Target         | Ratio             | The property's sale price in dollars.                      |
| MSSubClass       | Feature        | Nominal           | The building class.                                        |
| MSZoning         | Feature        | Nominal           | The general zoning classification.                        |
| LotFrontage      | Feature        | Ratio             | Linear feet of street connected to property.              |
| LotArea          | Feature        | Ratio             | Lot size in square feet.                                   |
| Street           | Feature        | Nominal           | Type of road access.                                       |
| Alley            | Feature        | Nominal           | Type of alley access.                                      |
| LotShape         | Feature        | Nominal           | General shape of property.                                |
| LandContour      | Feature        | Nominal           | Flatness of the property.                                 |
| Utilities        | Feature        | Nominal           | Type of utilities available.                              |
| LotConfig        | Feature        | Nominal           | Lot configuration.                                        |
| LandSlope        | Feature        | Nominal           | Slope of property.                                        |
| Neighborhood     | Feature        | Nominal           | Physical locations within Ames city limits.              |
| Condition1       | Feature        | Nominal           | Proximity to main road or railroad.                      |
| Condition2       | Feature        | Nominal           | Proximity to main road or railroad (if a second is present). |
| BldgType         | Feature        | Nominal           | Type of dwelling.                                         |
| HouseStyle       | Feature        | Nominal           | Style of dwelling.                                        |
| OverallQual      | Feature        | Ordinal           | Overall material and finish quality. 1 to 10, Very Poor to Very Excellent                     |
| OverallCond      | Feature        | Ordinal           | Overall condition rating. 1 to 10, Very Poor to Very Excellent                                 |
| YearBuilt        | Feature        | Interval          | Original construction date.                               |
| YearRemodAdd     | Feature        | Interval          | Remodel date.                                             |
| RoofStyle        | Feature        | Nominal           | Type of roof.                                             |
| RoofMatl         | Feature        | Nominal           | Roof material.                                            |
| Exterior1st      | Feature        | Nominal           | Exterior covering on house.                              |
| Exterior2nd      | Feature        | Nominal           | Exterior covering on house (if more than one material).  |
| MasVnrType       | Feature        | Nominal           | Masonry veneer type.                                      |
| MasVnrArea       | Feature        | Ratio             | Masonry veneer area in square feet.                      |
| ExterQual        | Feature        | Ordinal           | Exterior material quality.                               |
| ExterCond        | Feature        | Ordinal           | Present condition of the material on the exterior.       |
| Foundation       | Feature        | Nominal           | Type of foundation.                                       |
| BsmtQual         | Feature        | Ordinal           | Height of the basement. From NA (No Basement to Ex (100+ inches).                                 |
| BsmtCond         | Feature        | Ordinal           | General condition of the basement.                       |
| BsmtExposure     | Feature        | Nominal           | Walkout or garden level basement walls.                  |
| BsmtFinType1     | Feature        | Nominal           | Quality of basement finished area.                       |
| BsmtFinSF1       | Feature        | Ratio             | Type 1 finished square feet.                             |
| BsmtFinType2     | Feature        | Nominal           | Quality of second finished area (if present).            |
| BsmtFinSF2       | Feature        | Ratio             | Type 2 finished square feet.                             |
| BsmtUnfSF        | Feature        | Ratio             | Unfinished square feet of basement area.                 |
| TotalBsmtSF      | Feature        | Ratio             | Total square feet of basement area.                      |
| Heating          | Feature        | Nominal           | Type of heating.                                          |
| HeatingQC        | Feature        | Ordinal           | Heating quality and condition.                           |
| CentralAir       | Feature        | Binary           | Central air conditioning. N or Y                 |
| Electrical       | Feature        | Nominal           | Electrical system.                                        |
| 1stFlrSF         | Feature        | Ratio             | First Floor square feet.                                 |
| 2ndFlrSF         | Feature        | Ratio             | Second floor square feet.                                |
| LowQualFinSF     | Feature        | Ratio             | Low-quality finished square feet (all floors).           |
| GrLivArea        | Feature        | Ratio             | Above grade (ground) living area square feet.            |
| BsmtFullBath     | Feature        | Ratio             | Basement full bathrooms.                                 |
| BsmtHalfBath     | Feature        | Ratio             | Basement half bathrooms.                                 |
| FullBath         | Feature        | Ratio             | Full bathrooms above grade.                              |
| HalfBath         | Feature        | Ratio             | Half baths above grade.                                  |
| Bedroom          | Feature        | Ratio             | Number of bedrooms above basement level.                |
| Kitchen          | Feature        | Ratio             | Number of kitchens.                                      |
| KitchenQual      | Feature        | Ordinal           | Kitchen quality.                                         |
| TotRmsAbvGrd     | Feature        | Ratio             | Total rooms above grade (does not include bathrooms).    |
| Functional       | Feature        | Ordinal           | Home functionality rating.                               |
| Fireplaces       | Feature        | Ratio             | Number of fireplaces.                                    |
| FireplaceQu      | Feature        | Ordinal           | Fireplace quality.                                       |
| GarageType       | Feature        | Nominal           | Garage location.                                         |
| GarageYrBlt      | Feature        | Interval          | Year garage was built.                                   |
| GarageFinish     | Feature        | Nominal           | Interior finish of the garage.                          |
| GarageCars       | Feature        | Ratio             | Size of garage in car capacity.                         |
| GarageArea       | Feature        | Ratio             | Size of garage in square feet.                          |
| GarageQual       | Feature        | Ordinal           | Garage quality.                                          |
| GarageCond       | Feature        | Ordinal           | Garage condition.                                        |
| PavedDrive       | Feature        | Nominal           | Paved driveway.                                         |
| WoodDeckSF       | Feature        | Ratio             | Wood deck area in square feet.                          |
| OpenPorchSF      | Feature        | Ratio             | Open porch area in square feet.                         |
| EnclosedPorch    | Feature        | Ratio             | Enclosed porch area in square feet.                     |
| 3SsnPorch        | Feature        | Ratio             | Three season porch area in square feet.                 |
| ScreenPorch      | Feature        | Ratio             | Screen porch area in square feet.                       |
| PoolArea         | Feature        | Ratio             | Pool area in square feet.                               |
| PoolQC           | Feature        | Ordinal           | Pool quality.                                           |
| Fence            | Feature        | Ordinal           | Fence quality.                                          |
| MiscFeature      | Feature        | Nominal           | Miscellaneous feature not covered in other categories. |
| MiscVal          | Feature        | Ratio             | $Value of miscellaneous feature.                        
| MoSold           | Feature        | Ordinal           | Month Sold.                                             
| YrSold           | Feature        | Interval          | Year Sold.                                              
| SaleType         | Feature        | Nominal           | Type of sale.                                           
| SaleCondition    | Feature        | Nominal           | Condition of sale.                               

### Test Data
* **Source of Test Data**: Kaggle.com, [House Prices Prediction using TFDF](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/input)
* **Number of rows in test data**: 1459
* **State any differences in columns between training and test data**: Training data has labels, test data has no labels

### Model Details
* **Columns used as inputs in the final model:** 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'GarageArea', '1stFlrSF', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF'
* **Column(s) used as target(s) in the final model:** 'SalePrice'
* **Type of model:** XGBoost, LightGBM, ResNet Neural Networks
* **Software used to implement the model:** TensorFlow, XGBoost, LightGBM, Plotly, Pandas, NumPy, Scikit-learn, Resnet
* **Version of the modeling software:** TensorFlow v2.16.1, XGBoost 2.1.3, LightGBM 4.5.0, Plotly 5.24.1, Pandas 2.2.3, NumPy 2.2.0, Scikit-Learn 1.5.2, Resnet-152 
* **Hyperparameters or other settings of your model:**

ResNet Hyperparameters: 
```
if model_name == 'ResNet':
        model = create_tabular_resnet(X_train.shape[1])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        y_pred = model.predict(X_test).flatten()
        return model, y_pred, history
```
XGBoost Hyperparameters:
```
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, None
```
LightGBM Hyperparameters: 
```
    elif model_name == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, None
```
# Quantitative Analysis

* Models were assessed primarily with accuracy from the Kaggle Competition. See details below:

| **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** |
|--------------------|-------------------------|-------------------|
|RMSE:   0.1540         |RMSE: 0.1542         | RMSE:  0.15325      |

For Kaggle Submissions, predictions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. We calculated our training and validation accuracy with the same metrics as Kaggle's.

*Our submission prediction was the mean of XGboost and LightGBM predictions.

### Model Evaluation Metrics
 
| **Model**         | **Mean RMSE**    | **Std RMSE**    |
|--------------------|------------------|-----------------|
| Lasso             | $42,622.72       | $8,413.85       |
| Ridge             | $31,278.52       | $7,569.72       |
| Random Forest     | $30,066.71       | $4,084.11       |
| XGBoost           | $28,569.43       | $3,956.20       |
* XGBoost achieved the lowest RMSE on validation data, making it the most accurate model.

## Correlation Heatmap:
![The heatmap](https://github.com/xhu88/House-Prices-Project/blob/main/Heatmap.png)

Displays the relationships between all features and SalePrice.
Key Observations:
`OverallQual` and `GrLivArea` show the highest positive correlations with SalePrice.

* Top Correlated Features with SalePrice

| **Feature**       | **Correlation** |
|--------------------|-----------------|
| OverallQual        | 0.79            |
| GrLivArea          | 0.71            |
| GarageArea         | 0.62            |
| TotalBsmtSF        | 0.61            |
| YearBuilt          | 0.52            |

## Model Loss During Training:
![Model loss vis](https://github.com/xhu88/House-Prices-Project/blob/main/Model%20Loss%20During%20Training.png)

Training and validation loss stabilizes after 50 epochs.
Validation loss remains slightly lower than training loss, indicating good generalization.

## Residual Plot:
![Residual plot ](https://github.com/xhu88/House-Prices-Project/blob/main/Residual%20Plot.png)

Residuals are mostly centered around y=0, showing a good fit with slight heteroscedasticity.

## Real-World Implications and Actionable Insights     
* Model Reliability: The Model is effective for predicting house prices, particularly for mid-range properties, making it suitable for real estate applications.
* Key Features: Improve `OverallQual` and `GrLivArea` to maximize property value and command higher prices.
* Luxury Properties: The model needs to improve with high-priced, unique properties. Combine predictions with expert evaluations for accurate pricing.
* Market Analysis: Use the model to identify impactful features by neighborhood or region. Add specific features to refine or enhance it for luxury markets.
* Risk Management: Residual analysis highlights pricing risks for outlier properties. Use this to manage inconsistencies effectively.

# Ethical Considerations
* **Describe potential negative impacts of using your model**:
  * Math or software problems: 
The model has a risk of overfitting due to limited data, which can lead to issues in the model's implementation. Additionally, the model showed numerical instability when dealing with very large price values, which may lead to incorrect results. 
  * Real-World Risks: This model does not account for potential bias in features like Neighborhood and MSZoning, despite different neighborhoods and zoning often correlating with racial segregation, potentially accidentally factoring in race into pricing. Lastly, the dataset itself serves as a privacy concern, as individual houses from Ames can be identified from this data, which could be potentially used in targeted advertising or malicious campaigns. For these reasons and others not stated, this model should only be used for educational purposes.
* **Describe potential uncertainties relating to the impacts of using your model**:
  * Math or software problems: As the modeling group did not compile this dataset, meaning that it is unknown whether errors in the dataset, such as missing or imbalanced data, may propagate through the model, which may cause bias or errors in the output. Also, both the training and testing data may not represent all possible scenarios or populations, and the model's ability to generalize to novel scenarios is uncertain in this case.
  * Real-World Risks: This model is unlikely to be used again in any other scenario. The housing dataset is from Ames, Iowa, and the competition was hosted prior to 2016, so any use of this data outside of these limited contexts will lead to incorrect results. Additionally, any changes in market conditions and economic factors throughout when the data was captured were not recorded and could thus lead to inaccurate results.
* **Describe any unexpected results**: We ran into issues with having higher error rates for high-value properties despite our attempts at standardization. Additionally, the model performance was found to vary significantly across different property types, as some types were more abundant than others in the dataset.
