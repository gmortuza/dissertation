FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Install all required library
RUN pip install numpy~=1.19.2 \
                matplotlib~=3.3.2 \
                pyyaml~=5.3.1 \
                h5py~=2.10.0 \
                tqdm~=4.55.0 \
                scikit-learn~=0.24.1

#RUN mkdir /project
#WORKDIR /project

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
# && chown -R user:user /project
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Don't need to copy anything we will just use this environment for now
# COPY . /project

#RUN pip install -r requirements.txt

#CMD ["python3"]