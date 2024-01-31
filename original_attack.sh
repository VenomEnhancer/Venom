# nohup sh original_attack.sh > original.log 2>&1 &
models="vgg19_bn" # vgg19_bn preactresnet18
dataset="cifar10"
mode="clean"
device="cuda:0"
is_couple="False"
# badnet blended inputaware lc ssba trojannn wanet ada_blended
attack="badnet"
model="original_${attack}"
save_folder="${dataset}/${models}/original_attack/${model}"
mkdir -p record/${dataset}/${models}/original_attack/

if test ${attack} = "trojannn"
then
python ./attack/${attack}.py --is_couple ${is_couple} --model ${models} --sim_mode ${mode} --yaml_path ../config/attack/prototype/${dataset}.yaml --save_folder_name ${save_folder} --device ${device} --pretrain_model_path ../similarity/results/${dataset}/${models}/attack_result.pt --bd_yaml_path ../config/attack/trojannn/${models}.yaml
else
python ./attack/${attack}.py --is_couple ${is_couple} --model ${models} --sim_mode ${mode} --yaml_path ../config/attack/prototype/${dataset}.yaml --save_folder_name ${save_folder} --device ${device}
fi
# for defense in nad i-bau clp ft nc fp bnp npd
# do
# python ./defense/${defense}.py --model ${models} --result_file ${save_folder} --yaml_path ./config/defense/${defense}/${dataset}.yaml --dataset ${dataset}  --device ${device} 
# done