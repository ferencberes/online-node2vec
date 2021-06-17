#!/bin/bash
SAMPLES=1
THREADS=1
MAX_DAYS=1
#MAX_DAYS=15
python preprocess_data.py "../data"
python streamwalk_runner.py $SAMPLES $THREADS $MAX_DAYS
python evaluate.py "lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_square_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue" $SAMPLES $THREADS $MAX_DAYS
python second_order_runner.py $SAMPLES $THREADS $MAX_DAYS
python evaluate.py "lazy_decayedTrue-secondorder_hl43200_numh20_modhash200000_in0.00_out1.00_incrTrue-onlinew2v_dim128_lr0.0100_neg5_uratio0.80_square_mirrorTrue_omFalse_initgensim_expW1True_i86400_tnFalse_win0_pairsTrue" $SAMPLES $THREADS $MAX_DAYS