# Chatbot Intent Classification

This project creates a demo for an intent classification system to be used by a customer service chatbot.

All the data here was created using generative AI. 

Data manipulation and feature engineering was done using pandas, numpy and spacy. Model training, validation and experiment tracking was done using
sklearn and mlflow.

API was developed using FastAPI and Docker.

### Observations
- More samples in each single dataset will prevent overfitting.
- Datasets might be stored in binary format to avoid large file sizes
- Feature engineering is not necessary if using model such as BERT, but fine-tuning the model is a must with this approach for better results.
- Uvicorn is used for the API deployment but Gunicorn might give a better performance.
- Using an artifact storage will prevent redundant code and file duplication when doing the feature engineering and using the model in the API app. 