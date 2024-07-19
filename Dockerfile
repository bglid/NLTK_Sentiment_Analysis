FROM python:3.10

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV NAME SentimentAnalysisApp

CMD [ "streamlit", "run", "Streamlit_app.py" ]