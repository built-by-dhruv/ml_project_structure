FROM python:3.11-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install awscli -y

RUN pip install -r requirements.txt
CMD ["python", "app.py"]