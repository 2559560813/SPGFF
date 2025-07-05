
cd ../

TRAINER=SPGFF
OUTPUT=Your/Output/Path
DATA=Your/Data/Path

DATASET=$1
SEED=$2
SHOTS=$3

CFG=vit_b16_ep20


DIR=${OUTPUT}/SPGFF/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/SPGFF/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} 
fi

