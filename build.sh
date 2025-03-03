#!/bin/bash
HF_TOKEN=${1}
VERSION=${2}

echo "Building version: ${VERSION}"

# Dockerイメージのビルド（トークンを--build-argで渡す）
docker build --build-arg HF_TOKEN=${HF_TOKEN} -t yeq6x/sgen:${VERSION} .

# Dockerイメージのプッシュ
docker push yeq6x/sgen:${VERSION}