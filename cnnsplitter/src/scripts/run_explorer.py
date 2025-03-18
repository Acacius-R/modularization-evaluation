import subprocess
from multiprocessing import Pool
import sys
sys.path.append('..')
def run_instance(target_class):
    command = f"python ../module_explorer.py --model simcnn --dataset cifar100 --target_class {target_class}"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    target_classes = list(range(10))  # 0 to 9
    with Pool(processes=10) as pool:  # 10 parallel processes
        pool.map(run_instance, target_classes)
