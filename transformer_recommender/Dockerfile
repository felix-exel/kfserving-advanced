FROM python:3.7-slim

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "main.py"]
