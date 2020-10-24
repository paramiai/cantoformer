if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

apt-get install -y build-essential pkg-config opencc cmake doxygen
git clone https://github.com/BYVoid/OpenCC.git
cd OpenCC
git checkout ver.1.1.1
make
make install
cd ..
rm -r OpenCC
