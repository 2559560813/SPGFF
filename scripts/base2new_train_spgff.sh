
cd ../


TRAINER=SPGFF
OUTPUT=Your/Output/Path
DATA=Your/Data/Path

DATASET=$1
SEED=$2

CFG=vit_b16_b2n_ep10
SHOTS=16

DIR=${OUTPUT}/base2new/base/SPGFF/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base 
fi

