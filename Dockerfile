FROM python:3.7-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
CMD streamlit run application.py --server.port $PORT