ENV_NAME=InvertedPendulum-v2
WORLD_SIZE=4

mkdir sync
rm sync/*.pt
numactl --physcpubind=0 python main.py --task train --save_freq 1000 --rank 0 --display True --env-name $ENV_NAME &
numactl --physcpubind=1 python main.py --task train --save_freq 1000 --rank 1 --env-name $ENV_NAME &
numactl --physcpubind=2 python main.py --task train --save_freq 1000 --rank 2 --env-name $ENV_NAME &
numactl --physcpubind=3 python main.py --task train --save_freq 1000 --rank 3 --env-name $ENV_NAME &
python main.py --task optimize --save_freq 1000 --world_size $WORLD_SIZE --env-name $ENV_NAME &
wait
