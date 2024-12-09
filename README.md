# House Prices Model Card

### Basic Information

* **Group developing model**: Meghana Srinivasaiah, Jake Fitzpatrick, Jesse Mutamba, Xinya Hu
* **Model date**: December, 2024
* **Model version**: 1.0
* **License**: MIT
* **Model Implementation Code**:

### Intended Use
* **Primary Intended Uses**: This model is for educational uses only.
* **Primary Intended Users**: Students in DNSC 3288, and students learning about machine learning.
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

### Training Data

* **Source of Training Data**: Kaggle.com, House Prices Prediction using TFDF, https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/input
* **How training data was divided into test data and validation data**: 80/20 training - validation split. Test data is provided
* **Number of rows in training and validation data**:
  * Training rows:
  * Validation rows:

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
* **Source of Test Data**: Kaggle.com, House Prices Prediction using TFDF, https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/input
* **Number of rows in test data**:
* **State any differences in columns between training and test data**: Training data has labels, test data has no labels

### Model Details
* **Columns used as inputs in the final model:** 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'GarageArea', '1stFlrSF', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF'
* **Column(s) used as target(s) in the final model:** 'SalePrice'
* **Type of model:** Decision Tree
* **Software used to implement the model:** TensorFlow, XGBoost, LightGBM, Plotly, Pandas, NumPy, Scikit-learn
* **Version of the modeling software:** 
* **Hyperparameters or other settings of your model:**

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

    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, None

    elif model_name == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, None
```
# Quantitative Analysis

### Model Loss during training: 
* Training and validation loss steadly decrease steadly and stablitlize after 50 epochs.
* Validation loss reamains slightly lower than training loss. This indicates good generalization and no overfitting.
![Model loss vis](https://github.com/user-attachments/assets/dc44fa85-492d-474c-bbf0-b9313a88a4e1)


## Correclation Analysis
* A correclation heatmap was generated to highlight relationships among features and their impact on `sales price`.

| **Feature**       | **Correlation** |
|--------------------|-----------------|
| SalePrice         | 1.000           |
| OverallQual       | 0.791           |
| GrLivArea         | 0.709           |
| GarageCars        | 0.640           |
| GarageArea        | 0.623           |
| TotalBsmtSF       | 0.614           |
| 1stFlrSF          | 0.606           |
| FullBath          | 0.561           |
| BsmtQual_Ex       | 0.553           |
| TotRmsAbvGrd      | 0.534           |

* Strongest correlation( r= 0.791), emphasizing that the property's overall quality is the model's most influential feature. 
![Heatmap](https://github.com/user-attachments/assets/faf3df77-ab89-4e44-9122-38a5ce7e8edc)

# Feature Importance 

Key Insights:
* OverallQual: Most influential feature, contributing 60.4% to predictions.
* GrLivArea: Second most significant at 10.8%, reflecting the importance of above-ground living area.
* TotalBsmtSF: Adds 3.8%, showing the value of basement size in predicting house prices.

![Imporantce ](https://github.com/user-attachments/assets/095e4d9f-7195-475a-b6e0-d9d6cd76edc9)


## Residual Analysis 
* Residuals in the model are centered around y=0, indicating a good fit.
* Heteroscedasticity: Errors increase for higher predicted values(> 300,000).
* Outliers suggest overestimation for high-priced properties
![Residual plot ](https://github.com/user-attachments/assets/443b2c6b-4b23-4e81-bd1b-9ffa75cc9b8c)


### Ethical Considerations
* **Describe potential negative impacts of using your model**:
  * Math or software problems: Some of the potential negative impacts of using our model in the math/software aspect is that the errors in the dataset such as missing or imbalanced data may propagate through the model, which may cause bias or errors in the output. Additionally, if the random forest regressor is not properly tuned, such as set with incorrect hyperparameters, the model may perform well on the training data, but poorly on unseen data, which would also potentially lead to inaccurate predictions.
  * Real World Risks: Some potential real world risks is that if the predictions are used in high-stake scenarios, such as real estate pricing or financial forecasting, inaccurate outputs could lead to financial loss or unfair treatment of individuals. Over-reliance on model predictions without considering domain knowledge or expert input might also reusult in flawed decisions.
* **Describe potential uncertainties relating to the impacts of using your model**:
  * Math or software problems: AS a complex ensemble method, the Random Forest mmodel's predictions are difficult to interpret, potentially reducing trust, or leading to misuse. Also, both the training and testing data may not represent all possible scenarios or populations, which the model's ability to generalize to novel scenarios is uncertain in this case.
  * Real World Risks: Potential real-world uncertainties relating to the impacts of using our model is that disparties in the training data may result in unequal treatment of vulnerable populations in the model's outputs. If the training dataset lacks diversity or fails to capture real-world variability, predictions might not be reliable to real-world uses. Additionally, misinterpretation or overreliance on the model's results can perpetuate systemic inequalities or lead to unintended results.
* **Describe any unexpected or results**: Some issues that our group encountered was that our model experienced potentiall overfitting which would possibly cause variance in the data because we worked with a slightly imbalanced dataset. We also came across some of the common pre-processing issues such as imbalanced and missing data. Some of the other issuses are bugs and the prevention of black boxes, also making sure that our model does not convince and/or manipulate false information.
