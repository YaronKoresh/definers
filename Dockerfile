FROM python:3.11.1-slim

WORKDIR /

COPY builder/requirements-cuda.txt .
RUN pip install --no-cache-dir -r requirements-cuda.txt

COPY src/handler.py .

CMD ["python", "-u", "/handler.py"]