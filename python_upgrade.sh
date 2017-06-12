#Install required package
sudo apt-get install build-essential checkinstall 
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev


#Download python 3.5.2
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
sudo tar xzf Python-3.5.2.tgz

# Compile python source
cd Python-3.5.2 
sudo ./configure 
sudo make altinstall

# modify setup script


