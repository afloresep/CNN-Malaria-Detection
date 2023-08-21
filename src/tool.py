import os

def name(directory):
    ''''
    list directories not hidden (to avoid .DS_store and such when listing dir)
    '''
    dir = []
    for f in os.listdir(directory):
         if not f.startswith('.'):
            dir.append(f)
    return dir


def split_folder_to_train_test_valid(data_directory):
    print('hello')