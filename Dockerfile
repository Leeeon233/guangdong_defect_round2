FROM registry.cn-shenzhen.aliyuncs.com/leonzhao/guangdong_cloth:cuda
MAINTAINER leonzhao
ADD . /competition
WORKDIR /competition
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip --no-cache-dir install  -r requirements.txt
RUN pip install --no-cache-dir -e mmdetection/.
CMD ["sh", "run.sh"]