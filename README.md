# ames-housing-ml-project
Predicting house prices using machine learning
# Прогноз цен на недвижимость (Ames Housing)

Проект по машинному обучению (задача регрессии). Модель предсказывает стоимость домов в штате Айова с точностью **R² ≈ 0.947**.

Язык: Python
Библиотеки: Pandas, NumPy, SciPy (f_oneway), Scikit-Learn, XGBoost, Category Encoders

Что было сделано
Предобработка: Очистка данных, заполнение пропусков (включая медиану по районам).
Пайплайн: Масштабирование чисел (`StandardScaler`) и кодирование категорий (`TargetEncoder`) объединены в `ColumnTransformer`.
Моделирование: Настроен градиентный бустинг `XGBRegressor`, лучшие гиперпараметры подобраны через `GridSearchCV` на 5 фолдах.

Как запустить
1. Установить библиотеки: `pip install -r requirements.txt`
2. Запустить код: `python L1.py`
