#!/bin/bash
set -euo pipefail

EMB_DIM=${EMB_DIM:-64}
SEED=${SEED:-228}
AE_EPOCHS=${AE_EPOCHS:-200}
RISK_EPOCHS=${RISK_EPOCHS:-80}
UST_EPOCHS=${UST_EPOCHS:-80}
ANCHOR=${ANCHOR:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-0}
TEST_SIZE=${TEST_SIZE:-100}
SPLIT_SEED=${SPLIT_SEED:-$SEED}

SOURCE_CITY=${SOURCE_CITY:-singapore}
TARGET_CITY=${TARGET_CITY:-boston}
LINK_MODE=${LINK_MODE:-symlink}

RUN_BASELINES=${RUN_BASELINES:-1}
RUN_UST=${RUN_UST:-1}
RUN_EVAL=${RUN_EVAL:-1}
OVERWRITE_CITY_SPLIT=${OVERWRITE_CITY_SPLIT:-1}

CITY_PAIR="${SOURCE_CITY}_to_${TARGET_CITY}"
CITY_VIEW="data/NuPlan/city_views/${CITY_PAIR}"

SOURCE_MODEL="./models/City/BaselineB/${CITY_PAIR}/${SOURCE_CITY}_seed${SEED}_best_model.pt"
TARGET_MODEL="./models/City/BaselineB/${CITY_PAIR}/${TARGET_CITY}_seed${SEED}_best_model.pt"
UST_MODEL="./models/City/UST/${CITY_PAIR}/anchor${ANCHOR}/${SOURCE_CITY}_${TARGET_CITY}_anchor${ANCHOR}_seed${SEED}_best_model.pt"

SOURCE_AE_MODEL="${SOURCE_MODEL/_best_model.pt/_ae_best_model.pt}"
TARGET_AE_MODEL="${TARGET_MODEL/_best_model.pt/_ae_best_model.pt}"

split_args=(
	--source_city "$SOURCE_CITY"
	--target_city "$TARGET_CITY"
	--test_size "$TEST_SIZE"
	--split_seed "$SPLIT_SEED"
	--link_mode "$LINK_MODE"
)

if [ "$OVERWRITE_CITY_SPLIT" = "1" ]; then
	split_args+=(--overwrite)
fi

echo "Preparing clean city datasets: source=${SOURCE_CITY}, target=${TARGET_CITY}"
python -m scripts.city_split_processing "${split_args[@]}"

if [ "$RUN_BASELINES" = "1" ]; then
	echo -e "\nRunning BaselineB source-city training (${SOURCE_CITY})"
	python -m scripts.city_train baseline \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--domain source \
	--train_autoencoder \
	--train_risk \
	--seed "$SEED" \
	--embed_dim "$EMB_DIM" \
	--best_model_path "$SOURCE_MODEL" \
	--ae_epochs "$AE_EPOCHS" \
	--risk_epochs "$RISK_EPOCHS" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS"

	echo -e "\nRunning BaselineB target-city training (${TARGET_CITY})"
	python -m scripts.city_train baseline \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--domain target \
	--train_autoencoder \
	--train_risk \
	--seed "$SEED" \
	--embed_dim "$EMB_DIM" \
	--best_model_path "$TARGET_MODEL" \
	--ae_epochs "$AE_EPOCHS" \
	--risk_epochs "$RISK_EPOCHS" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS"
fi

if [ "$RUN_UST" = "1" ]; then
	echo -e "\nRunning City UST (${SOURCE_CITY} -> ${TARGET_CITY}) with anchor ${ANCHOR}%"
	python -m scripts.city_train ust \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--anchor_pct "$ANCHOR" \
	--load_best_ae_clean \
	--ae_clean_ckpt_path "$SOURCE_AE_MODEL" \
	--load_best_ae_noisy \
	--ae_noisy_ckpt_path "$TARGET_AE_MODEL" \
	--train_stage2 \
	--stage2_epochs "$UST_EPOCHS" \
	--stage2_lr 1e-4 \
	--seed "$SEED" \
	--embed_dim "$EMB_DIM" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS" \
	--best_model_path "$UST_MODEL"
fi

if [ "$RUN_EVAL" = "1" ]; then
	echo -e "\n+++++++++++++++++++ Evaluating city models +++++++++++++++++++++++"

	echo -e "\nEvaluating BaselineB source-city model (${SOURCE_CITY})"
	python -m scripts.city_train baseline \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--domain source \
	--evaluate \
	--best_model_path "$SOURCE_MODEL" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS"

	echo -e "\nEvaluating BaselineB target-city model (${TARGET_CITY})"
	python -m scripts.city_train baseline \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--domain target \
	--evaluate \
	--best_model_path "$TARGET_MODEL" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS"

	echo -e "\nEvaluating City UST model (${SOURCE_CITY} -> ${TARGET_CITY})"
	python -m scripts.city_train ust \
	--source_city "$SOURCE_CITY" \
	--target_city "$TARGET_CITY" \
	--city_view_root "$CITY_VIEW" \
	--anchor_pct "$ANCHOR" \
	--evaluate \
	--best_model_path "$UST_MODEL" \
	--batch_size "$BATCH_SIZE" \
	--num_workers "$NUM_WORKERS"
fi
