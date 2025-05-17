# Install required dependencies
sudo apt install g++ python3 python3-dev pkg-config sqlite3 cmake libgsl-dev

# Download and extract NS-3
wget -c https://www.nsnam.org/releases/ns-allinone-3.35.tar.bz2
tar -xvjf ns-allinone-3.35.tar.bz2
cd ns-allinone-3.35/ns-3.35/

# Add missing header include to affected files
echo "#include <cstdint>" | cat - src/network/utils/bit-serializer.h > temp && mv temp src/network/utils/bit-serializer.h
echo "#include <cstdint>" | cat - src/network/utils/bit-deserializer.h > temp && mv temp src/network/utils/bit-deserializer.h
echo "#include <cstdint>" | cat - src/wifi/model/block-ack-type.h > temp && mv temp src/wifi/model/block-ack-type.h

# Configure and build with Python bindings disabled
./waf configure --enable-examples --disable-python
./waf
./waf --run hello-simulator

cd ~ && cd workspace/car/ns-allinone-3.35/netanim-3.108/

sudo apt install qtchooser

sudo apt install -y qtcreator qtbase5-dev qt5-qmake cmake

sudo apt-get install qt6-base-dev

qmake NetAnim.pro

make

./NetAnim