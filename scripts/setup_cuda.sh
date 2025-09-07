#!/bin/bash

mkdir /etc/modprobe.d

chmod a+rwx /tmp

apt-get update
apt-get install -y curl dkms

apt-get purge -y nvidia-*
apt-get autoremove -y

cat > /etc/modprobe.d/blacklist-nouveau.conf <<EOF
blacklist nouveau
options nouveau modeset=0
EOF

update-initramfs -u

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb

apt-get update
apt-get install -y cuda-toolkit

CU_PATHS=("/usr/local/cuda-12/" "/usr/local/cuda/" "/opt/cuda/")
LD_PATHS=("/usr/local/cuda-12/lib64/" "/usr/local/cuda/lib64/" "/opt/cuda/lib64/")

CU_PATH=""
LD_PATH=""

for path in "${CU_PATHS[@]}"; do
    if [ -d "${path}" ]; then
        CU_PATH="${path}"
        echo "Found CUDA Path: $CU_PATH"
        break
    fi
done

for path in "${LD_PATHS[@]}"; do
    if [ -d "${path}" ]; then
        LD_PATH="${path}"
        echo "Found LD_LIBRARY_PATH: $LD_PATH"
        break
    fi
done

if [ -z "$CU_PATH" ] || [ -z "$LD_PATH" ]; then
    echo "Error: CUDA installation not found in expected directories." >&2
    exit 1
fi

export CUDA_PATH="$CU_PATH"
export LD_LIBRARY_PATH="$LD_PATH"
export PATH="${CU_PATH}bin:$PATH"

echo "CUDA_PATH set to: $CUDA_PATH"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
