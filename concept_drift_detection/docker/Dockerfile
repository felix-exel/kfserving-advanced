FROM python:3.7-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc build-essential libgl1-mesa-glx libglib2.0-0

COPY . .
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

ENTRYPOINT ["python", "-u", "main.py"]
