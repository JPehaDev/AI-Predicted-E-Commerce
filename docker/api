FROM tiangolo/uvicorn-gunicorn:python3.8-slim

WORKDIR /app

ADD requirements.txt .

RUN pip install --upgrade -r requirements.txt \
    && rm -rf /root/.cache

COPY . .

# Expose the port on which the application will run
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]