FROM python:3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata sudo

RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python \
    python3-dev \
    python3-pip \
    ffmpeg libsm6 libxext6 iputils-ping\
  && apt-get clean

RUN sudo pip3 install Flask  matplotlib numpy opencv-python python-telegram-bot requests scikit-image scipy pycocotools nltk gunicorn shortuuid
RUN sudo pip3 install opencv-contrib-python
ADD  src/ /opt/api/
WORKDIR /opt/api/
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "entrypoint.sh"]
