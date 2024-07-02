#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This scripts takes a docker image that already contains the MaxText dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_maxtext_jax_ss_image_upload.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Example command:
# bash docker_maxtext_jax_ss_image_upload.sh PROJECT_ID=sample_project BASEIMAGE=gcr.io/sample_project/jax-ss/tpu:jax0.4.28-rev1.0.0 CLOUD_IMAGE_NAME=maxtext-jax-ss-0.4.28-rev1.0.0 IMAGE_TAG=latest

set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ ! -v PROJECT_ID ]]; then
  echo "Erroring out because PROJECT is unset, please set it!"
  exit 1
fi

if [[ ! -v CLOUD_IMAGE_NAME ]]; then
  echo "Erroring out because CLOUD_IMAGE_NAME is unset, please set it!"
  exit 1
fi

if [[ ! -v IMAGE_TAG ]]; then
  echo "Erroring out because IMAGE_TAG is unset, please set it!"
  exit 1
fi

COMMIT_HASH=$(git rev-parse --short HEAD)

echo "Building JAX SS MaxText at commit hash ${COMMIT_HASH} . . ."  

docker build \
  --build-arg JAX_SS_BASEIMAGE=${BASEIMAGE} \
  --build-arg COMMIT_HASH=${COMMIT_HASH} \
  --network=host \
  -t gcr.io/${PROJECT_ID}/${CLOUD_IMAGE_NAME}:${IMAGE_TAG} \
  -f ./maxtext_jax_ss_tpu.Dockerfile .

docker push gcr.io/${PROJECT_ID}/${CLOUD_IMAGE_NAME}:${IMAGE_TAG}

echo "All done, check out your artifacts at: gcr.io/${PROJECT_ID}/${CLOUD_IMAGE_NAME}:${IMAGE_TAG}"