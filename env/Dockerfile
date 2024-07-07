# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Install SSH and other necessary packages
RUN apt-get update && apt-get install -y \
    openssh-server \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up SSH
RUN mkdir /var/run/sshd
RUN echo 'root:ss' | chpasswd # Change password to much secure one!!
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

# Expose the SSH port
EXPOSE 22

# Set the default command to execute
# when creating a new container
CMD ["/usr/sbin/sshd", "-D"]
