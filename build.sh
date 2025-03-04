#!/bin/bash

# # BuildKitを有効化
# export DOCKER_BUILDKIT=1

HF_TOKEN=${1}
VERSION=${2}

echo "Building version: ${VERSION}"

# Dockerイメージのビルド（キャッシュを使用）
docker build --build-arg HF_TOKEN=${HF_TOKEN} -t yeq6x/sgen:${VERSION} .

# Dockerイメージのプッシュ
docker push yeq6x/sgen:${VERSION}