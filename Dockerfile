FROM registry.cn-shenzhen.aliyuncs.com/leonzhao/guangdong_cloth:se_finetune
MAINTAINER leonzhao
ADD . /competition
WORKDIR /competition
ENV CUDA_HOME /usr/local/cuda-10.0
ENV PATH $PATH:/usr/local/cuda-10.0/bin
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64
#RUN apt-get update && apt-get install -y libglib2.0-0 libglib2.0-dev libsm6 libxrender-dev libxext6 \
# && apt-get clean \
# && rm -rf /var/lib/apt/lists/*
#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip --no-cache-dir install  -r requirements.txt
#RUN chmod -R 755 /usr/local/cuda-10.0/
#RUN pip install --no-cache-dir -e /competition/mmdetection/.
CMD ["sh", "run.sh"]