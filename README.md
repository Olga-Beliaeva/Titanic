# Titanic
Due to introduce myself through a project
# Project Titanic
- первый проект, задуман как базовый проект для портфолио 
- данные взяты из Kaggle
- данные для обучения разделены на 2 части в соотношении 75/25 (обучение/валидация)
- для оценки моделей использованы данные для валидации
- выбирались модели по лучшим результатам: f1_score, accuracy, roc_auc
- результаты проверялись через Kaggle public score

 **I.PREPROCESSING DATA**
- заполнены пропущенные данные
- сделана базовая аналитика

**II.FEATURES' DESIGN**
- отобраны и подготовлены признаки
- данные подготовлены к обучению: числовые данные стандартизованы, катигориальные данные закодированы

**III. SUPERVISED LEARNING**
- отобраны модели,  гиперпараметры подоброны при помощи GridSearchCV

**RendomForestClassification**

![image](https://user-images.githubusercontent.com/56861356/146897479-6f1ef9c1-3843-4a24-b81e-da7dfa6c3094.png)
 
  - **best public score = 0.78468** 
 
   - features= ['Sex', 'Pclass', 'Fare']
    
**VotingClassifier**

![image](https://user-images.githubusercontent.com/56861356/146897703-380616d8-83c1-4b5c-ba03-09dc93bab4cf.png)
  
  - **best public score = 0.77033**
  
  - features=['Sex', 'Embarked', 'Family2', 'Age_cat2']	
