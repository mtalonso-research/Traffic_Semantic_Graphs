#!/bin/bash
EMB_DIM=64
SEED=228 # 3 seeds I used: 42, 3407, 1234
AE_EPOCHS=200
RISK_EPOCHS=80
UST_EPOCHS=80
ANCHOR=5
noises=(35)

# Train clean model 
# AE: 200 epochs
echo "Running Clean Training"
# python -m scripts.4A_ae_risk \
# --nup \
# --clean \
# --train_autoencoder \
# --train_risk \
# --seed "$SEED" \
# --embed_dim "$EMB_DIM" \
# --best_model_path ./models/BaselineB/clean/clean_seed"$SEED"_best_model.pt \
# --ae_epochs 200 \
# --risk_epochs "$RISK_EPOCHS" \
# --batch_size 16 \
# --num_workers 0

# Run Noisy Training and UST Training per noise level
for n in "${noises[@]}"; do
	# Generate Noisy data for noise level n
		
	# Check if the noisy data for this noise level already exists, if not generate it
	if [ ! -d "data/NuPlan/training_data/noisy_$n" ]; then
		echo -e "\nGenerating Noisy train data for noise level $n"
		mkdir -p data/NuPlan/training_data/noisy_"$n"

		python scripts/1C_noise_processing.py \
		--data_dir data/NuPlan/training_data/clean/graphs \
		--output_dir data/NuPlan/training_data/noisy_"$n"/graphs \
		--noise_level "$n"
		echo -e "~Waiting 20 seconds to let filesystem catch up..."
		sleep 20

		python -m scripts.2_risk_analysis \
		--run_on_all_episodes \
		--input_directory data/NuPlan/training_data/noisy_"$n"/graphs \
		--output_directory data/NuPlan/training_data/noisy_"$n" \
		--output_filename risk_scores
		echo -e "~Waiting 20 seconds to let filesystem catch up..."
		sleep 20
	fi
	if [ ! -d "data/NuPlan/evaluation_data/noisy_$n" ]; then
		echo -e "\nGenerating Noisy evaluation data for noise level $n"
		mkdir -p data/NuPlan/evaluation_data/noisy_"$n"

		python scripts/1C_noise_processing.py \
		--data_dir data/NuPlan/evaluation_data/clean/graphs \
		--output_dir data/NuPlan/evaluation_data/noisy_"$n"/graphs \
		--noise_level "$n"
		echo -e "~Waiting 20 seconds to let filesystem catch up..."
		sleep 20

		python -m scripts.2_risk_analysis \
		--run_on_all_episodes \
		--input_directory data/NuPlan/evaluation_data/noisy_"$n"/graphs \
		--output_directory data/NuPlan/evaluation_data/noisy_"$n" \
		--output_filename risk_scores_true
		echo -e "~Waiting 20 seconds to let filesystem catch up..."
		sleep 20
	fi

	# Noisy
	echo -e "\nRunning Noisy Training for noise level $n"

	python -m scripts.4A_ae_risk \
	--nup \
	--noisy "$n" \
	--train_autoencoder \
	--train_risk \
	--seed "$SEED" \
	--embed_dim "$EMB_DIM" \
	--best_model_path ./models/BaselineB/noisy"$n"/noisy"$n"_seed"$SEED"_best_model.pt \
	--ae_epochs "$AE_EPOCHS" \
	--risk_epochs "$RISK_EPOCHS" \
	--batch_size 16 \
	--num_workers 0
	echo -e "~Waiting 20 seconds to let filesystem catch up..."
	sleep 20
	
	# UST
	echo -e "\nRunning UST Training for noise level $n and anchor $ANCHOR"
	python scripts/5A_ust_risk.py \
	--nup \
	--clean "$ANCHOR" \
	--noisy "$n" \
	--load_best_ae_clean \
	--ae_clean_ckpt_path ./models/BaselineB/clean/clean_seed"$SEED"_ae_best_model.pt \
	--load_best_ae_noisy \
	--ae_noisy_ckpt_path ./models/BaselineB/noisy"$n"/noisy"$n"_seed"$SEED"_ae_best_model.pt \
	--train_stage2 --stage2_epochs "$UST_EPOCHS" \
	--stage2_lr 1e-4 \
	--seed "$SEED" \
	--embed_dim "$EMB_DIM" \
	--batch_size 16 \
	--num_workers 0 \
	--best_model_path ./models/UST/noisy"$n"/anchor"$ANCHOR"/clean"$ANCHOR"_noisy"$n"_seed"$SEED"_best_model.pt
	echo -e "~Waiting 20 seconds to let filesystem catch up..."
	sleep 20
done


# Run Evaluate of all models to form a readable output
echo -e "+++++++++++++++++++ Evaluating all models +++++++++++++++++++++++"
for n in "${noises[@]}"; do
	# Clean
	echo -e "\nEvaluating Clean model"
	python -m scripts.4A_ae_risk \
	--nup \
	--clean \
	--evaluate \
	--best_model_path ./models/BaselineB/clean/clean_seed"$SEED"_best_model.pt \
	--batch_size 16 \
	--num_workers 0
	
	# Noisy
	echo -e "\nEvaluating Noisy model for noise level $n"
	python -m scripts.4A_ae_risk \
	--nup \
	--noisy "$n" \
	--evaluate \
	--best_model_path ./models/BaselineB/noisy"$n"/noisy"$n"_seed"$SEED"_best_model.pt \
	--batch_size 16 \
	--num_workers 0

	# UST
	echo -e "\nEvaluating UST model for noise level $n and anchor $ANCHOR"
	python scripts/5A_ust_risk.py \
	--nup \
	--clean "$ANCHOR" \
	--noisy "$n" \
	--evaluate \
	--best_model_path ./models/UST/noisy"$n"/anchor"$ANCHOR"/clean"$ANCHOR"_noisy"$n"_seed"$SEED"_best_model.pt \
	--batch_size 16 \
	--num_workers 0
done
