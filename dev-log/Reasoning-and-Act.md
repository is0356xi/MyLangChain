# 実行環境準備

## Windowsから直接使う場合
- 参考
  - https://qiita.com/motoyuki1963/items/a334c9488c2f55a867cf

- cuDNNとCUDA-Tool-Kitをダウンロードする
  - [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)から```v8.9.1-for-CUDA11.x```をダウンロード
  - [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)から```11.8.0```をダウンロード

- パスを通す
  - cuDNN
    - ダウンロードして展開したフォルダ下の```bin```のパスを通す
  - CUDA Tool Kit
    - システム環境変数に```CUDA_PATH```が追加されているのを確認し、
    - ユーザ環境変数に```%CUDA_PATH%*```で必要なパスを通す

```sh:
# cuDNN
C:\path\to\cudnn\bin

# CUDA Took-Kit
%CUDA_PATH%bin
%CUDA_PATH%libnvvp
%CUDA_PATH%lib64
%CUDA_PATH%include
```

- pipで必要なパッケージをインストール
  - https://pytorch.org/
  - 上記サイトにアクセスし、自身の環境を入力すると実行すべきコマンドが表示される


```sh:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dockerを使う場合
- 参考
  - https://qiita.com/komiya_5467/items/eb3174ffbd410acdbd6a
  - https://developer.nvidia.com/rdp/cudnn-download
  - https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
  
- MetaのGithubにあるLLaMAを試す
  - https://github.com/facebookresearch/llama
  - メモリ不足でできず、、。他のモデルに変更
  - https://github.com/nomic-ai/gpt4all

```yaml:
version: '3.9'
services:
  cuda:
    image: nvidia/cuda:11.8.0-devel-ubuntu20.04
    environment:
      - CUDNN_VERSION=8.9.1.23

    # リソースを拡張したい場合の指定（任意）
    shm_size: 16gb
    ulimits:
      memlock: -1
      stack: 67108864

    # GPUを有効にする（docker-compose 1.28.0以上で対応）
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

    # 永続起動on
    tty: true

    # 実行に必要なものをコンテナ上にマウントする
    volumes:
      - C:\Users\hoge\hoge\Serverless-App\ReAct\:/home/
      - C:\Users\hoge\hoge\Serverless-App\ReAct\llama:/home/llama
      - D:\LLaMa\:/Downloads/
```

- 起動したコンテナに入って必要なものをインストール

```sh:
# コンテナに入る
docker exec -it $(docker ps -q -n 1) bash

# cudnnの環境構築
apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.8 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.8 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

# pythonの環境構築
apt update && apt install -y python3-pip && pip install --upgrade pip
cd /home && pip install -r requirements.txt

# cudaが使えるかテストする
/usr/bin/python3.8 test_cuda.py

# イメージを保存しておく
exit
docker commit <コンテナID> cuda-11.8 
```

------------------------------------------------------------

# サンプル実行

```sh:
/usr/bin/python3.8 sample.py

# Out of Memoryエラーが発生
```

------------------------------------------------------------

# llama.ccpに変更してみる
- https://github.com/ggerganov/llama.cpp


```sh:
git clone https://github.com/ggerganov/llama.cpp
```

- Windowsでmakeするために、w64devkitをダウンロード
  - https://github.com/skeeto/w64devkit/releases

```sh:
./w64devkit
cd llama.cpp
make
```

- BLASを使用するために、OpenBLAS for Winをダウンロード
  - https://github.com/xianyi/OpenBLAS/releases

```sh:
# aファイルとincludeフォルダをコピー
cp .\OpenBLAS-0.3.23-x64\lib\libopenblas.a .\w64devkit\x86_64-w64-mingw32\lib\
cp .\OpenBLAS-0.3.23-x64\include\* .\w64devkit\x86_64-w64-mingw32\include\  

# w64devkitで再びmake
cd llama.cpp
make LLAMA_OPENBLAS=1
```

- modelのダウンロード後、binに変換

```sh:
# binに変換
py -m pip install -r requirements.txt
py convert.py ./models/7B

# quantize
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0
```


- 実行確認

```sh:
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512
```

-----------------------------

# 本家に統合できるか
- モデルのパラメータファイルのサイズが13GBから4GBになった
  - Out of Memoryエラーがなくなるかも？
- できなさそうなので、Pythonライブラリのtransformersを使用しているOpenLLMに変更する

----------------------------------


# GPT4All
- https://github.com/nomic-ai/gpt4all
  - 盛り上がっていそう。約40kのスター

## 環境構築

- bin形式のPre-Trainedモデルをダウンロード
  - https://gpt4all.io/models/ggml-gpt4all-j.bin


```sh:
# gpt4allのソースコード
git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git
cd gpt4all && git submodule update --init

# GPU Interface
git clone https://github.com/nomic-ai/nomic.git
```

- cudaコンテナ起動

```sh:
# python3.8へパスを通す
apt install -y vim && vi ~/.bashrc
#### 追記 ####
export PATH=/usr/bin:$PATH
alias python='python3.8'
#############
source ~/.bashrc

# pythonパッケージインストール
cd /Downloads/gpt4all
py -m pip install -r requirements.txt

cd ../nomic
py -m pip install .[GPT4ALL]
py -m pip install nomic

# テスト
cd ../gpt4all
python gpu_test.py   
# OSError: It looks like the config file at 'models/ggml-gpt4all-j.bin' is not a valid JSON file.が発生
```


## 原因調査

- https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
  - ここの588~675を見てみる
- https://huggingface.co/transformers/v4.4.2/model_doc/auto.html


```py:
pretrained_model_name_or_path = str(pretrained_model_name_or_path)

is_local = os.path.isdir(pretrained_model_name_or_path)
if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
    # Special case when pretrained_model_name_or_path is a local file
    resolved_config_file = pretrained_model_name_or_path
    is_local = True
```

--------------------------------------------------

# LangChainに組み込む

- まとめ
  - https://qiita.com/sakue_103/items/72cd06a2cdd723437eef

- 参考
  - https://python.langchain.com/en/latest/modules/models/llms/integrations/gpt4all.html
  - https://python.langchain.com/en/latest/getting_started/getting_started.html
  - https://github.com/nomic-ai/gpt4all/issues/470


---------------------------------

# Transformers

- 参考
  - https://tt-tsukumochi.com/archives/4908