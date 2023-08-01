# **Puzzles 3D**

Puzzles 3D is a dataset specifically designed for 3D puzzle assembly tasks. It consists of a collection of 3D puzzles that need to be assembled by matching and aligning their constituent pieces. The dataset includes 6 different 3D puzzles, each represented by a set of fragmented pieces.

Here is the [link](https://www.geometrie.tuwien.ac.at/ig/3dpuzzles.html) to download the dataset.

To begin, we need to create the "geometry" directory if it does not already exist.

~~~
mkdir geometry
cd geometry
~~~

After downloading the dataset, the directory structure should look as follows:

~~~
└── geometry
    └── puzzles3d
        ├── brick
        ├── cake
        └── ...
~~~