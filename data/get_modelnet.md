# **ModelNet40**

ModelNet40 is a widely recognized benchmark dataset for 3D object recognition and classification. It contains a diverse collection of 12,311 CAD models belonging to 40 different object categories, ranging from everyday objects to vehicles and furniture. To evaluate the effectiveness of our proposed reduced DGCNN model, we conduct experiments on the ModelNet40 dataset. To obtain this dataset, you can follow the steps outlined below for downloading it.

To begin, we need to create the "extractor" directory if it does not already exist.

~~~
mkdir extractor
~~~

Then, we download the dataset.

~~~
cd extractor
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
unzip modelnet40_ply_hdf5_2048.zip
mv modelnet40_ply_hdf5_2048 modelnet40
rm modelnet40_ply_hdf5_2048.zip
~~~
