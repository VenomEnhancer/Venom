# nohup sh defense.sh > defense.log 2>&1 &
device="cuda:0"
dataset="cifar10"
models="vgg19_bn"
save_folder="cifar10/vgg19_bn/original_attack/original_badnet"
defense="npd" # nad i-bau clp ft nc fp bnp npd
python ./defense/${defense}.py --model ${models} --result_file ${save_folder} --yaml_path ./config/defense/${defense}/${dataset}.yaml --dataset ${dataset}  --device ${device} 