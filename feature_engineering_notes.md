The following ipynb file has feature engineering and data preparation 
https://github.com/thomas-tf/MSBD5001GroupProject/blob/priyanko/feature-engineering-and-data-preparation-5001gp.ipynb

Note -
1. Only columns with certain properties - a)One unique value b)More than 90% values are NULL c)One value covers more than 90% of records - are dropped
2. Columns other than Top columns from RFE are not dropped - but the code is commented you can use that
3. RFE is given in - https://github.com/thomas-tf/MSBD5001GroupProject/blob/priyanko/recursive-feature-elimination.ipynb
4. Some new Aggregation columns created - like mean and std on Transaction Amount vs Card1 etc. These features are chosen from RFE study.
5. Label encoding was initially done on Category columns - but I later changed that to Columns as object. If you want to encode categorical columns we have make list properly.
6. NaaN values are not handled - based on your model handle that
