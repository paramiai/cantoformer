## Commands

### Sync Data from Google bucket for codes

```bash
USER=XXX
BUCKET_NAME=XXX
REPO_NAME=generate_data
ROOT=/home/$USER
REPO=$ROOT/$REPO_NAME
mkdir $REPO
gsutil rsync -r gs://$BUCKET_NAME/gits/$REPO_NAME $REPO

```

### Mount Data

```bash
# Show disks available
sudo lsblk

# Mount read-only
sudo mount -o ro,noload /dev/sdb  /mnt/d

# Mount read-write
sudo mount -o discard,defaults /dev/sdb /mnt/d


```

### Pre-training

#### TPU
```bash
# Activate Google VM torch-xla-nightly conda
conda activate torch-xla-nightly


export TPU_IP_ADDRESS=XXX

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1
export TPU_NAME=node-1

TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=10000
UPDATE_FREQ=1
BATCH_SIZE=16 # per gpu
SEQ_LENGTH=256
LR=5e-4
MODEL=electra_v2
MODEL_SIZE=small
VOCAB_FILE=cantokenizer-vocab.txt
DATA_DIR=/mnt/d/
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH"_1M"


python train.py $DATA_DIR \
    --model $MODEL \
    --save-dir "$SAVE_DIR" \
    --log-interval 1000 \
    --weight-decay 0.01 \
    --seq-length $SEQ_LENGTH \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --num-workers=1 \
    --model-size=$MODEL_SIZE \
    --vocab-file=$VOCAB_FILE \
    --gen-ratio=4 \
    --tpu \
    --cooked \
    --disc-weight=50

```

#### GPU
```bash
TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=10000
UPDATE_FREQ=1
BATCH_SIZE=16 # per gpu
SEQ_LENGTH=256
LR=5e-4
MODEL=electra_v1
MODEL_SIZE=small
VOCAB_FILE=cantokenizer-vocab.txt
DATA_DIR=/mnt/d/
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH
python train.py \
    $DATA_DIR \
    --model $MODEL \
    --model-size=$MODEL_SIZE \
    --save-dir "$SAVE_DIR" \
    --log-interval 100 \
    --weight-decay 0.01 \
    --seq-length $SEQ_LENGTH \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --num-workers=2 \
    --vocab-file=$VOCAB_FILE \
    --gen-ratio=4 \
    --disc-weight=50 \
    --cooked

```



### Finetuning




#### TPU
```bash
# Activate Google VM torch-xla-nightly conda
conda activate torch-xla-nightly


export TPU_IP_ADDRESS=XXX

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1
export TPU_NAME=node-1


TOTAL_NUM_UPDATES=2
WARMUP_UPDATES=0.06
UPDATE_FREQ=1
BATCH_SIZE=2 # per gpu
SEQ_LENGTH=256
LR=2e-5
MODEL=electra_v1
MODEL_SIZE=small
VOCAB_FILE=cantokenizer-vocab.txt
DATA_DIR=/tmp/mnli/
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH
python train.py \
    $DATA_DIR \
    --model $MODEL \
    --save-dir "$SAVE_DIR" \
    --log-interval 100 \
    --weight-decay 0.01 \
    --seq-length $SEQ_LENGTH \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --num-workers=0 \
    --model-size=$MODEL_SIZE \
    --vocab-file=$VOCAB_FILE \
    --gen-ratio=4 \
    --disc-weight=50 \
    --tpu \
    --cooked

```


#### GPU
```bash
TOTAL_NUM_UPDATES=2
WARMUP_UPDATES=0.06
UPDATE_FREQ=1
BATCH_SIZE=2 # per gpu
SEQ_LENGTH=256
LR=2e-5
MODEL=electra_v1
MODEL_SIZE=small
VOCAB_FILE=cantokenizer-vocab.txt
DATA_DIR=/tmp/mnli/
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH
python train.py \
    $DATA_DIR \
    --model $MODEL \
    --save-dir "$SAVE_DIR" \
    --log-interval 100 \
    --weight-decay 0.01 \
    --seq-length $SEQ_LENGTH \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --num-workers=0 \
    --model-size=$MODEL_SIZE \
    --vocab-file=$VOCAB_FILE \
    --gen-ratio=4 \
    --disc-weight=50 \
    --cooked

```
