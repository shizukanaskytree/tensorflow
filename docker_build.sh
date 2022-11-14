# 聪明的方法: https://www.tensorflow.org/install/docker

# 1. NVIDIA Container Toolkit
### Docker-CE on Ubuntu can be setup using Docker’s official convenience script:
curl https://get.docker.com | sh && sudo systemctl --now enable docker

# DEPRECATION WARNING
#     This Linux distribution (ubuntu xenial) reached end-of-life and is no longer supported by this script.
#     No updates or security fixes will be released for this distribution, and users are recommended
#     to upgrade to a currently maintained version of ubuntu.

### Setup the package repository and the GPG key:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list



