# CarlaFSD

The package for [Carla](!http://carla.org/) simulation of self-driving car.

Note: All instructions below is given for Ubuntu 18.04 or 20.04.


# Installation
First create carla workspace folder e.g.:
```bash
mkdir ~/carla_workspace
cd carla_workspace
```
Than clone `CarlaFSD` repository into this folder.

### Python environment

1. Open linux shell and install virtualenv as following:
```bash
sudo apt install virtualenv
sudo apt install  python3.7
```

2. Create python3 virtualenv inide `carla_workspace` folder as follows (tested with python 3.7):
```bash
virtualenv ~/carla_workspace/carla_venv -p python3.7
```
3. Activate virtualenv
```bash
source ~/carla_workspace/carla_venv/bin/activate
```
4. Install necessary requirements
```bash
cd ~/carla_workspace/CarlaFSD
pip install -r requirements.txt
```

5. If you want to deactivate virtualenvironment use:
```bash
deactivate
```

### Carla installation

Go to the page with [releases](https://github.com/carla-simulator/carla/releases/) and upload latest release `tar` archive for Linux and additional maps into `~/carla_workspace/archive` folder.
1. Create floder to extract:
```bash
mkdir -p ~/carla_workspace/CARLA
```
2. Unpack tar archive (tested for carla 0.9.15 version):
```bash
tar -xvzf ~/carla_workspace/archive/CARLA_0.9.15.tar.gz -C ~/carla_workspace/CARLA
``` 
3. Move the package with additional maps into `~/carla_workspace/CARLA/Import` and unpack as follows:
```bash
mv ~/carla_workspace/archive/AdditionalMaps_0.9.14.tar.gz ~/carla_workspace/CARLA/Import/
cd ~/carla_workspace/CARLA
bash ImportAssets.sh
```
4. Finally install carla into python environment:
```
pip install ~/carla_workspace/CARLA/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
pip install -r  ~/carla_workspace/CARLA/PythonAPI/examples/requirements.txt
```

### Git lfs installation

Install git lfs:
```bash
cd ~/carla_workspace/CarlaFSD
~/carla_workspace/CarlaFSD
```

# Run CARLA

Before strating any script below first run carla simulator by the following command:
```bash
cd ~/carla_workspace/CARLA

bash CarlaUE4.sh

```
This command should run simulator server.

## Run fisheye drive

In order to run fisheye camera drive apply command:

```bash
source ~/carla_workspace/carla_venv/bin/activate
cd ~/carla_workspace/CarlaFSD
python -m carla_fsd.camera_fisheye.drive
```
Note: since we are not building package the run command should be invoked from `CarlaFSD` folder.