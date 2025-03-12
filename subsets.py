import os
import shutil
import random
import pathlib


def n_jobs_count(n_jobs):
    cpu_num = [i for i in range(os.cpu_count()+1)]
    if n_jobs > 0 or n_jobs < 0:
        n_jobs = cpu_num[n_jobs]
    else:
        raise ValueError(f"n_jobs must be an integer from 1 to {os.cpu_count()} or from -1 to -{os.cpu_count()}.")
    
    return n_jobs


def subset_dirs(src, label):
    return os.path.join(pathlib.Path(src).parents[0], label)


# creates subsets for both CPU multiprocessing and validation sets
def create_subsets(src, label, sample_size=5000, n_jobs=1):
    
    n_jobs = n_jobs_count(n_jobs)

    dest = subset_dirs(src, label)
    # checks to see which tiles are already in the destination folder
    dest_list = []
    dest_gtr = os.walk(dest)
    layer = 1
    for (_, _, filenames) in dest_gtr:
        if layer == 3:
            dest_list.extend(filenames)
            break
        layer+=1
    
    # randomly selects tiles from dataset, and places them into subset folders
    success = 0
    folder_counter = 1
    while success < sample_size and sample_size < len(os.listdir(src)):
        file = random.choice(os.listdir(src))
        if file not in dest_list:
            if os.path.exists(os.path.join(dest, str(folder_counter))) == False:
                os.makedirs(os.path.join(dest, str(folder_counter)))
            shutil.copy(os.path.join(src, file), os.path.join(dest, str(folder_counter)))
            if folder_counter == n_jobs:
                folder_counter = 1
            else:
                folder_counter+=1
            dest_list.append(file)
            success += 1

    return dest
