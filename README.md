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

* **Source of Training Data**:
* **How training data was divided into test data and validation data**:
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
| OverallQual      | Feature        | Ordinal           | Overall material and finish quality.                     |
| OverallCond      | Feature        | Ordinal           | Overall condition rating.                                 |
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
| BsmtQual         | Feature        | Ordinal           | Height of the basement.                                  |
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
| CentralAir       | Feature        | Nominal           | Central air conditioning.                                |
| Electrical       | Feature        | Nominal           | Electrical system.                                        |
| 1stFlrSF         | Feature        | Ratio             | First Floor square feet.                                 |
| 2ndFlrSF         | Feature        | Ratio             | Second floor square feet.                                |
| LowQualFinSF     | Feature        | Ratio             | Low quality finished square feet (all floors).           |
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
