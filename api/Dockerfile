FROM    python:3.11.4-slim
LABEL   decription="Intent Classification API for a Chatbot "
LABEL   version="1.0"
WORKDIR /app
COPY    . /app
RUN     apt-get update && apt-get install -y gdal-bin libgdal-dev
RUN     pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000