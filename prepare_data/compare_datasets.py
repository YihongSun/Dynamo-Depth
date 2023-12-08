import filecmp
import os, sys
import os.path as osp
from tqdm import tqdm

def get_all_files(directory):
    all_files = []
    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            # Get the full path of the file
            file_path = os.path.join(dirpath[len(directory):], filename)
            # Append the file path to the list
            all_files.append(file_path)
    return all_files

def compare_directories(dir1, dir2):
    dir1_files = get_all_files(dir1)
    dir2_files = get_all_files(dir2)
    common_files = set(dir1_files) & set(dir2_files)

    print(f'# of files in {dir1}: {len(dir1_files)}')
    print(f'# of files in {dir2}: {len(dir2_files)}')
    print(f'# of common files: {len(common_files)}')

    all_good = True
    for file in tqdm(common_files, 'Comparing (Error will be printed right away)'):
        f1 = osp.join(dir1, file)
        f2 = osp.join(dir2, file)

        if not osp.exists(f1):
            print(f'\n### Error! {f1} does not exist!')
            all_good = False
            continue
        if not osp.exists(f2):
            print(f'\n### Error! {f2} does not exist!')
            all_good = False
            continue
        if not filecmp.cmp(f1, f2):
            print(f'\n### Error! {file} is different in two directories!')
            all_good = False
    if all_good:
        print('All common files matched!')

compare_directories(sys.argv[1], sys.argv[2])