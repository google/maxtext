ARG JAX_SS_BASEIMAGE

# JAX Stable Stack Base Image
From $JAX_SS_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

# Install Python packages from requirements.txt
RUN pip install -r /deps/requirements.txt

# Run the script available in JAX-SS base image to generate the manifest file
RUN bash /generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH