# nohup sh venom_attack.sh > test.log 2>&1 &
models="vgg19_bn" # vgg19_bn preactresnet18
dataset="cifar10" # cifar10 cifar100 gtsrb
device="cuda:0"
sim_beta=20
test_mode="venom"
mode="single_deep_conv_10"
half_epochs=5
test_name="${mode}_beta_${sim_beta}_half_${half_epochs}"
is_couple="True"

mkdir -p record/${dataset}/${models}/${test_mode}/${test_name}

# badnet trojannn blended ssba wanet inputaware lc ada_blended
attack="badnet"
model="$venom_{attack}"
save_folder="${dataset}/${models}/${test_mode}/${test_name}/${model}"
if test ${attack} = "trojannn"
then
python ./attack/${attack}.py --half_epochs ${half_epochs} --model ${models} --sim_beta ${sim_beta} --is_couple ${is_couple} --sim_mode ${mode} --yaml_path ../config/attack/prototype/${dataset}.yaml --save_folder_name ${save_folder} --device ${device} --pretrain_model_path ../similarity/results/${dataset}/${models}/attack_result.pt --bd_yaml_path ../config/attack/trojannn/${models}.yaml
else
python ./attack/${attack}.py --half_epochs ${half_epochs} --model ${models} --sim_beta ${sim_beta} --is_couple ${is_couple} --sim_mode ${mode} --yaml_path ../config/attack/prototype/${dataset}.yaml --save_folder_name ${save_folder} --device ${device}
fi
# for defense in nad i-bau clp ft nc fp bnp npd
# do
# python ./defense/${defense}.py --model ${models} --result_file ${save_folder} --yaml_path ./config/defense/${defense}/${dataset}.yaml --dataset ${dataset}  --device ${device} 
# done
