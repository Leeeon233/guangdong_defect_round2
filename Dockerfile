FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3
MAINTAINER leonzhao
ADD . /competition
WORKDIR /competition
RUN pip --no-cache-dir install  -r requirements.txt
CMD ["sh", "run.sh"]