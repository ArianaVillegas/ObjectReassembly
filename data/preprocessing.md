# **Preprocessing**

In order to standardize the number of points to 2048 and generate the labels for the matching step, the Breaking Bad and Puzzles 3D datasets undergo preprocessing. This preprocessing step is essential to ensure consistency in the input data and facilitate the subsequent matching analysis.

To activate the environment, run
~~~
conda env export > environment.yml
conda activate torch
~~~

For preprocessing, run
~~~
python utils/gen_labels.py --subset=SUBSET_NAME
~~~

SUBSET_NAME = [artifact, everyday, puzzles3d]