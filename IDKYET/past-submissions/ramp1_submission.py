import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from liverrisk.model import LiverriskModel, score_model

class MyLiverRiskModel(LiverriskModel):
    def __init__(self):
        self.feature_columns = None
        self.hepatic_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.death_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def process_inputs(self, X):
        X_processed = X.copy()
        #First I get the numeric and categorical columns
        numeric_cols = X_processed.select_dtypes(include=['number']).columns
        categorical_cols = X_processed.select_dtypes(exclude=['number']).columns
        #Then I fill the missing values for numeric columns with the median
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        #For categorical columns, i fill with "missing"
        X_processed[categorical_cols] = X_processed[categorical_cols].fillna("missing")

        #Then I encode the categorical variables using one-hot encoding, this 
        #is because most models cant direcctly handle categorical variables names
        #like male or female. By converting them to 0 and 1, the model can understand them better.
        #so instead of sex: male female or missing, we now have a column for sex_Female (which can be 0,1), 
        #sex_male and sex_ missing
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols)

        # When it runs for the first time, it will set the feature columns to the cols of the processed data.
        if self.feature_columns is None:
            self.feature_columns = X_processed.columns
        #After the first time, this code will run
        #It forces the new dataframe to have the same columns as the one used for training, 
        # filling missing columns with 0
        # Maybe during training I had [age, sex_male, sex_female, sex_missing]
        #But during prediction I only have [age, sex_male]. This would be a problem because the model expects the same columns as during training.
        #By reindexing the columns to match the training columns and filling missing ones with 0, I ensure that the model can still make predictions 
        # without errors.
        else:
            X_processed = X_processed.reindex(columns=self.feature_columns, fill_value=0)

        return X_processed
        

    def fit(self, X, y):
        X_processed = self.process_inputs(X)
        y_hepatic = y["evenements_hepatiques_majeurs"].fillna(0)  
        y_death = y["death"].fillna(0)

        self.hepatic_model.fit(X_processed, y_hepatic)
        self.death_model.fit(X_processed, y_death)

        return self

    def predict(self, X):
        X_processed = self.process_inputs(X)

        hepatic_pred = self.hepatic_model.predict(X_processed)
        death_pred = self.death_model.predict(X_processed)

        return pd.DataFrame({
            "risk_hepatic_event": hepatic_pred, 
            "risk_death": death_pred
        }, index=X.index)  

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    model = MyLiverRiskModel()
    score = score_model(model, args.data_dir)
    print(f"Score: {score:.4f}")