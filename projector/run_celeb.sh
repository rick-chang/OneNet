IMG_SIZE=64
TRIAL=1
PRETRAIN_RECON=0
BATCH_SIZE=50  # reference batch included
N_REFERENCE=25
DPERIOD=1
GPERIOD=1
LEARNING_RATE_PROJ=2e-4
LEARNING_RATE_DIS=2e-4     # 2e-4 in context encoder, 2e-4 in dcgan, 5e-5 in wgan
WEIGHT_DECAY_RATE=0 #1e-6      
LAMBDA_RATIO=1e-2
LAMBDA_L2=5e-3
LAMBDA_LATENT=1e-4
LAMBDA_IMG=1e-3
LAMBDA_DE=1.0
DE_DECAY_RATE=1.0
SOFT_POS=0.85
CONT_NOISE=1
NOISE_STD=0.5
USE_SPATIAL_VARYING_NOISE=1
UNIFORM_NOISE_MAX=1.732     # 1.732 (sqrt(3) to retain the std) # (wrong) 3.464 (sqrt(12) to retain the standard deviation)
MIN_SPATIALLY_CONTINOUS_NOISE_FACTOR=0.01
MAX_SPATIALLY_CONTINOUS_NOISE_FACTOR=0.1
PRETRAIN_ITER=0
ADAM_BETA1_D=0.5            # 0.5 in all the papers
ADAM_BETA2_D=0.999
ADAM_EPS_D=1e-8
ADAM_BETA1_G=0.9            # 0.5 in all the papers
ADAM_BETA2_G=0.999
ADAM_EPS_G=1e-6
BASE_FOLDER='model'
USE_TENSORBOARD=1
TENSORBOARD_PERIOD=30
OUTPUT_IMG=0
OUTPUT_IMG_PERIOD=200
CLIP_INPUT=0
CLIP_INPUT_BOUND=10.0
CLAMP_WEIGHT=1
CLAMP_LOWER=-10.0
CLAMP_UPPER=10.0


STD_FOLDER='std_outputs/'${BASE_FOLDER}

if [ ! -d 'std_outputs' ]; then
  mkdir 'std_outputs'
fi

if [ ! -d ${STD_FOLDER} ]; then
  mkdir ${STD_FOLDER}
fi

STD_FOLDER+='/'ratio${LAMBDA_RATIO}_dis${LAMBDA_L2}_latent${LAMBDA_LATENT}_img${LAMBDA_IMG}_de${LAMBDA_DE}_derate${DE_DECAY_RATE}_dp${DPERIOD}_gd${GPERIOD}_softpos${SOFT_POS}

if [ ! -d ${STD_FOLDER} ]; then
  mkdir ${STD_FOLDER}
fi

SCRIPT_NAME=${0##*/}


python -u main.py \
    --img_size $IMG_SIZE \
    --Dperiod $DPERIOD \
    --Gperiod $GPERIOD \
    --clamp_lower $CLAMP_LOWER \
    --clamp_upper $CLAMP_UPPER \
    --clamp_weight $CLAMP_WEIGHT \
    --batch_size $BATCH_SIZE \
    --n_reference $N_REFERENCE \
    --learning_rate_val_proj $LEARNING_RATE_PROJ \
    --learning_rate_val_dis $LEARNING_RATE_DIS\
    --weight_decay_rate $WEIGHT_DECAY_RATE \
    --one_sided_label_smooth $SOFT_POS \
    --lambda_ratio $LAMBDA_RATIO \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_latent $LAMBDA_LATENT \
    --lambda_img $LAMBDA_IMG \
    --lambda_de $LAMBDA_DE \
    --de_decay_rate $DE_DECAY_RATE \
    --noise_std $NOISE_STD \
    --continuous_noise $CONT_NOISE \
    --use_spatially_varying_uniform_on_top $USE_SPATIAL_VARYING_NOISE \
    --uniform_noise_max $UNIFORM_NOISE_MAX \
    --min_spatially_continuous_noise_factor $MIN_SPATIALLY_CONTINOUS_NOISE_FACTOR \
    --max_spatially_continuous_noise_factor $MAX_SPATIALLY_CONTINOUS_NOISE_FACTOR \
    --adam_beta1_d $ADAM_BETA1_D \
    --adam_beta2_d $ADAM_BETA2_D \
    --adam_eps_d $ADAM_EPS_D \
    --adam_beta1_g $ADAM_BETA1_G \
    --adam_beta2_g $ADAM_BETA2_G \
    --adam_eps_g $ADAM_EPS_G \
    --base_folder $BASE_FOLDER \
    --pretrained_iter $PRETRAIN_ITER \
    --use_tensorboard $USE_TENSORBOARD \
    --tensorboard_period $TENSORBOARD_PERIOD \
    --output_img $OUTPUT_IMG \
    --output_img_period $OUTPUT_IMG_PERIOD \
    --clip_input $CLIP_INPUT \
    --clip_input_bound $CLIP_INPUT_BOUND \
    > >(tee ${STD_FOLDER}/${SCRIPT_NAME}_trail${TRIAL}.out) \
    2> >(tee ${STD_FOLDER}/${SCRIPT_NAME}_trail${TRIAL}.err >&2)
