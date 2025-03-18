import os



models = ['fc3','fc5']
datasets = ['mnist','kmnist','fmnist']

for model in models:
    for dataset in datasets:
      cmd = f'python modularize.py ' \
        f'--model {model} --dataset {dataset} >> log.txt'
      print(cmd)
      os.system(cmd)