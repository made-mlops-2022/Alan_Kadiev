FROM python:3.8.13-slim-buster

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY online_model.py /online_model.py

ENV MODEL_URL = "https://drive.google.com/file/d/1ZRoxFOganPtsEe0ijfSaw9tkm_T2m30C/view?usp=sharing"

CMD ["uvicorn", "online_model:app", "--host", "0.0.0.0", "--port", "8080"]