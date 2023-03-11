FROM pytorch/pytorch
WORKDIR /RCTF_supercat
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY static static
COPY templates templates
COPY run.py run.py
COPY flag.py flag.py
ENTRYPOINT ["python", "run.py"]
# docker run -p 0.0.0.0:8888:5000 catspy
# docker build -t catspy .