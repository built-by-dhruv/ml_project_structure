FROM python:3.11-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install awscli -y

RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080


CMD ["python", "app.py"]