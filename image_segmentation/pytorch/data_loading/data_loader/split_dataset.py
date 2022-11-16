from sklearn.model_selection import KFold
import tensorflow.io as io
# importing random module
import random
import os
import re
def load_data(path, files_pattern):
    data = sorted(io.gfile.glob((os.path.join(path, files_pattern))))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def fold_split(path: str, fold_num : int):
    imgs = load_data(path, "*_x.npy")
    split = KFold(n_splits=fold_num, random_state=random.seed(3), shuffle=True)
    
    for idx, item in enumerate(split.split(imgs)):
        with open('brats_evaluation_cases_{}.txt'.format(idx), 'w') as f:
            for i in item[1]:
                filename = imgs[i]
                f.write(re.split('/', filename)[-1].split('_')[1])
                f.write('\n')

if __name__ == "__main__":
    fold_split("gs://mlperf-dataset/data/2021_Brats_np/11_3d", 5)