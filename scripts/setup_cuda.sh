#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo."
    exit
fi

mkdir /usr/share/keyrings/
mkdir /etc/modprobe.d

chmod -R a+rwx /tmp
chmod -R a+rwx /usr/bin
chmod -R a+rwx /usr/lib
chmod -R a+rwx /usr/local

apt-get update
apt-get install -y curl

export PATH=/sbin:$PATH

apt-get update
apt-get purge nvidia-*

echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf

apt-get install -y --reinstall dkms
apt-get install -f

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb

cd /usr/share/keyrings/

ar vx cuda.deb
tar xvf data.tar.xz

mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg

rm -r /usr/share/keyrings/usr/
rm -r /usr/share/keyrings/etc/

echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list

chmod a+rwx /usr/share/keyrings/cuda-archive-keyring.gpg
chmod a+rwx /etc/apt/sources.list.d/CUDA.list

apt-get update
apt-get install -y cuda-toolkit

CU_PATHS=("/opt/cuda*/" "/usr/local/cuda*/")
LD_PATHS=("/opt/cuda*/lib" "/usr/local/cuda*/lib" "/opt/cuda*/lib64" "/usr/local/cuda*/lib64")

CU_PATH=""
LD_PATH=""

for path in "${CU_PATHS[@]}"; do
    if compgen -G "${path}" > /dev/null; then
        CU_PATH=$(ls -d "${path}" | head -n 1)
        break
    fi
done

for path in "${LD_PATHS[@]}"; do
    if compgen -G "${path}" > /dev/null; then
        LD_PATH=$(ls -d "${path}" | head -n 1)
        break
    fi
done

if [ -n "$CU_PATH" ] && [ -n "$LD_PATH" ]; then
    export CUDA_PATH="$CU_PATH"
    export LD_LIBRARY_PATH="$LD_PATH"
    echo "CUDA_PATH set to: $CUDA_PATH"
    echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
else
    echo "Error: CUDA not found or library path is missing." >&2
    exit 1
fi
