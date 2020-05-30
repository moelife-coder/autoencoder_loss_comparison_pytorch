cp_result() {
    if [ ! -d "${1}_${2}" ]
    then
        mkdir "${1}_${2}"
    fi
    mv train.log "${1}_${2}/${3}.log"
    mv result.pkl "${1}_${2}/${3}.pkl"
}
source run.sh
for i in mlp cnn
do
    for j in mnist cifar
    do
        for k in mse l1 ssim psnr l1s
        do
            if [ "${i}" == "mlp" ]
            then
                if [ "${j}" == "mnist" ]
                then
                    res=392
                else
                    res=1536
                fi
            else
                if [ "${j}" == "mnist" ]
                then
                    res=2
                else
                    res=6
                fi
            fi
            echo "training ${i} ${j} ${k}"
            train_${i}_${j} ${res} ${k} result.pkl
            cp_result ${i} ${j} ${k}
        done
    done
done
tar cf ../packed.tar ./
