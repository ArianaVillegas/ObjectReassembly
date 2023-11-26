models=("pointnet" "pointnet2" "dgcnn" "vn_pointnet" "vn_dgcnn" "ours" "vn_ours")

for val1 in ${models[*]}; do
    echo $val1
    python feature_extractor.py --cfg_file="config/extractor/${val1}.yml"
done