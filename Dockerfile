FROM python:3.8

COPY zeroshot-classify/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt && rm -rf /tmp/requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"
WORKDIR /abb/scripts

ADD scripts /abb/scripts

ENV TRANSFORMERS_OFFLINE='1'
ENV HF_DATASETS_OFFLINE='1'