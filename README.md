# **Object Reassembly**
Created by Ariana Villegas.

Welcome to the Object Reassembly repository! This repository provides an implementation of an object reassembly pipeline, consisting of two main steps: matching and alignment. In order to perform these steps, a feature extractor is required. To address this, we propose a reduced version of the Dynamic Graph CNN (DGCNN) as our feature extractor. This reduced DGCNN enables efficient and effective feature extraction from point cloud data. Our pipeline encompasses all these steps, allowing for the reassembly of objects from various fragmented parts.

## **0. Create environment**
~~~
conda env export > environment.yml
~~~

## **1. Extractor training**
~~~
python train_extractor.py
~~~
