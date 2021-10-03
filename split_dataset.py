import splitfolders

path_to_data = "../dataset"
path_to_output = "split"
seed = 1337
ratio = (0.8, 0.1, 0.1)


def split_data():
    """Split dataset using splitfolders package into three folders: train, val, test"""
    splitfolders.ratio(path_to_data, output=path_to_output, seed=seed, ratio=ratio)
    print("Split successful")


split_data()
