import os
import sys
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.description='please enter np file path ...'
    parser.add_argument("-n", "--np", help="numpy to print", dest="np")
    args = parser.parse_args()

    np_file = args.np
    print("numpy file is :", np_file)

    np_data = np.load(np_file)
    print(f'numpy_data : {np_data.shape}')
    print(f'numpy_data : {np_data}')
