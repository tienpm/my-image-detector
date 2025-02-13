#! /bin/bash

conda env create -n tf_env -f ./conda/tf_env.yml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
ln -s $CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/libnvinfer.so.8 $CONDA_PREFIX/lib/libnvinfer.so.7
ln -s $CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/libnvinfer_plugin.so.8 $CONDA_PREFIX/lib/libnvinfer_plugin.so.7
pip instal -r ./tf_requirements.txt

# Add kernel to Jupyter Lab
python -m ipykernel install --user --name=tf_env
# Trigger Jupyter Lab server
# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &!