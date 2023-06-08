# models=("pointnet" "pointnet2" "dgcnn" "vn_pointnet" "vn_dgcnn" "ours")
models=("pointnet2" "dgcnn" "vn_pointnet" "ours")

for val1 in ${models[*]}; do
    echo $val1
    python train_extractor.py --cfg_file="config/extractor/train_extractor_${val1}.yml"
done