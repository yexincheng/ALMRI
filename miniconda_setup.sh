# get miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
$HOME/miniconda3/bin/conda init bash
$HOME/miniconda3/bin/conda init zsh
