# **Breaking Bad**

Breaking Bad dataset is a diverse collection of 3D objects, including everyday items, vehicles, furniture, and cultural artifacts. It provides a rich resource for training and evaluating models in 3D object recognition, classification, and reconstruction. With its varied content and realistic details, the dataset enables researchers to develop innovative algorithms and solutions for real-world challenges in computer vision. 

Here is the [link](https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io/blob/main/README.md) of the official documentation to download Artifact and Everyday subset.

To begin, we need to create the "geometry" directory if it does not already exist.

~~~
mkdir geometry
cd geometry
~~~

After downloading the dataset, the directory structure should look as follows:

~~~
└── geometry
    ├── artifact
    │   ├── 39085_sf
    │   ├── 39086_sf
    │   └── ...
    └── everyday
        ├── BeerBottle_3f91158956ad7db0322747720d7d37e8
        ├── BeerBottle_6da7fa9722b2a12d195232a03d04563a
        └── ...
~~~