module load anaconda3/

source $HOME/Conda_setting.txt

conda create -n galaxy python=3.10
conda activate galaxy
conda install pip
pip install pandas numpy scipy statsmodels scikit-learn seaborn dask
