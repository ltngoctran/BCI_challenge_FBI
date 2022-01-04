import os
from sys import platform

# Folders
project_name = 'BCI-MVA'
project_dir  = os.path.dirname(os.path.dirname(__file__))
results_dir  = project_dir + '/results/'
data_dir     = project_dir + '/data/'
fit_dir      = project_dir + '/fit/'
fit1_dir     = project_dir + '/fit1/'
ext ='.csv'


#dir_path = os.path.dirname(os.path.realpath(__file__)) # src
print(data_dir)
print(results_dir)
