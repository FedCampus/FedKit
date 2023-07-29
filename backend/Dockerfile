FROM python:3.11.3-slim-bullseye
COPY ./ /app/

RUN python3 -m pip install -r /app/requirements.txt

RUN python3 /app/manage.py migrate

CMD sh /app/run.sh