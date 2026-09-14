import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from category_encoders import TargetEncoder as taren

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

df = pd.read_csv(r"C:\Users\User\Desktop\housing.csv")



# Очистка и подготовка данных



nan_col = ["Garage Type","Garage Finish","Garage Qual","Garage Cond","Bsmt Qual","Bsmt Cond","Bsmt Exposure","BsmtFin Type 1","BsmtFin Type 2","Fireplace Qu","Mas Vnr Type"]
for col in nan_col:
    df[col] = df[col].fillna("None")

nan_num_col = ["BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF","Total Bsmt SF","Bsmt Full Bath","Bsmt Half Bath","Mas Vnr Area"]
for col in nan_num_col:
    df[col] = df[col].fillna(0)

df = df.drop(columns=["Unnamed: 0","Order","PID","Alley","Pool QC","Fence","Misc Feature"])
df = df.dropna(subset=["Electrical","Garage Cars","Garage Area"]).reset_index(drop=True)
df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])

print(df.info())
print(df.isnull().sum())
print(df.sample(10))
print(df.nunique())



# Подготовка признаков и Train/Test



x = df.drop(columns=["SalePrice"]).copy()
y = df["SalePrice"].astype(float).copy()

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)

medi = x_train.groupby("Neighborhood")["Lot Frontage"].median()
x_train["Lot Frontage"] = x_train["Lot Frontage"].fillna(x_train["Neighborhood"].map(medi))
x_test["Lot Frontage"] = x_test["Lot Frontage"].fillna(x_test["Neighborhood"].map(medi))



# Фильтрация признаков



odj_col = x_train.select_dtypes(include="object").columns
obj_col_drop = []

for col in odj_col:
    if x_train[col].value_counts(normalize=True).iloc[0] > 0.95:
        obj_col_drop.append(col)

x_train = x_train.drop(columns=obj_col_drop)
x_test = x_test.drop(columns=obj_col_drop)


obj_col2 = x_train.select_dtypes(include="object").columns
obj_col2_drop = []

for col in obj_col2:
    cat = [y_train[x_train[col] == value].values
        for value in x_train[col].unique()]
    if len(cat) > 1:
        f_stat, p_value = f_oneway(*cat)
        if p_value > 0.05:
            obj_col2_drop.append(col)
    else:
        obj_col2_drop.append(col)

print(obj_col2_drop)



# Preprocessing и модель


num = x_train.select_dtypes(include="number").columns
cat = x_train.select_dtypes(include="object").columns

prep = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num),
    ("cat", taren(), cat)])

pipe = Pipeline(steps=[
    ("preprocessor", prep),
    ("model", xgb.XGBRegressor(objective="reg:squarederror",random_state=42))])



# Обучение и оценка модели



pipe.fit(x_train, y_train)

pred = pipe.predict(x_test)

print("R^2", r2_score(y_test, pred))
print("MAE", mean_absolute_error(y_test, pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, pred)))



# Анализ важности признаков



z = pipe["model"].feature_importances_
x = pipe["preprocessor"].get_feature_names_out()
f_i = pd.DataFrame({"Feature": x,"Importance": z})
f_i = f_i.sort_values(by="Importance",ascending=False).reset_index(drop=True)

print(f_i)



# Подбор гипер-параметров

param_grid = {
    "model__n_estimators": [1000, 1100, 1200, 1300],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
    "model__subsample": [ 0.4, 0.5, 0.6, 0.7],
    "model__colsample_bytree": [0.3, 0.4, 0.5],
}

#0.942
#0.946
#0.947

grid = GridSearchCV(pipe,param_grid=param_grid,cv=5,scoring="r2",n_jobs=-1)
grid.fit(x_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(x_test)

print("R²:", r2_score(y_test, pred))
print(grid.best_params_)
















