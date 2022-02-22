# Setup Jetson boot from SSD

## Preparing the Host
* ubuntu 18.04 host is needed.     
* a USB cable(speed is importanat)    
* Jetson SSD already installed    
* internet connection for your host(faster better)    
```bash
#git clone the jetson hack repo
git clone https://github.com/jetsonhacks/bootFromExternalStorage.git

#open the repo
cd bootFromExternalStorage

# install the dep.
./install_dependencies.sh
# get the jetson dev file from NV(~15min)
./get_jetson_files.sh
```

## Flashing the Jetson
* get the Jetson power into Force Recovery mode
* get the Jetson power into Force Recovery mode
* get the Jetson power into Force Recovery mode
Then enter the following line on the host   
```bash
bash flash_jetson_external_storage.sh
```

## power on the Jetson
setup ros py3 on jetson side stuff like this

