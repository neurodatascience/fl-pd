
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source config/.env

# pick PREVENT-AD imaging sessions to use
./scripts/choose_pad_imaging_sessions.py $FPATH_PAD_MANIFEST $FPATH_PAD_MCI $DPATH_FL_DATA

# create single large TSV for dach dataset
./scripts/get_data-adni.py $FPATH_ADNI_PHENO $FPATH_ADNI_ASEG $FPATH_ADNI_APARC
./scripts/get_data-calgary.py $FPATH_CALGARY_PHENO $FPATH_CALGARY_ASEG $FPATH_CALGARY_APARC
./scripts/get_data-pad.py $FPATH_PAD_DEMOGRAPHICS $FPATH_PAD_AGE $FPATH_PAD_MCI $FPATH_PAD_ASEG $FPATH_PAD_APARC $DPATH_FL_DATA
./scripts/get_data-ppmi.py $FPATH_PPMI_PHENO $FPATH_PPMI_ASEG $FPATH_PPMI_APARC --fs7 --fpath-aseg-fs6 $FPATH_PPMI_ASEG_FS6 --fpath-aparc-fs6 $FPATH_PPMI_APARC_FS6
./scripts/get_data-qpn.py $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_ASEG $FPATH_QPN_APARC $DPATH_FL_DATA

# extract columns
./scripts/subset_data.py --dropna COG_DECLINE --decline --age --sex --no-diag --cases --controls --aparc --no-aseg $DPATH_FL_DATA_LATEST {adni,calgary,pad,ppmi,qpn}
./scripts/subset_data.py --dropna AGE --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $DPATH_FL_DATA_LATEST {adni,calgary,pad,ppmi,qpn}
# ./scripts/subset_data.py --dropna DIAGNOSIS --no-decline --age --sex --diag --cases --controls --aparc --aseg $DPATH_FL_DATA_LATEST {adni,calgary,ppmi,qpn}

# # also get the entire control subset
# ./scripts/subset_data.py --dropna AGE --dropna SEX --decline --age --sex --diag --no-cases --controls --aparc --aseg $DPATH_FL_DATA_LATEST {adni,ppmi,qpn}

# # simulated (MMSE)
# ./scripts/get_data-simulated.py $DPATH_FL_DATA

# split into training and testing sets
# no normative model (also no z-scoring)
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,calgary,pad,ppmi,qpn}-decline-age-sex-case-hc-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,calgary,pad,ppmi,qpn}-age-sex-hc-aseg
# ./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --min-age 55 --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn,pad}-age-sex-hc-aseg
# ./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
# no normative model but with z-scoring
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,calgary,pad,ppmi,qpn}-decline-age-sex-case-hc-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,calgary,pad,ppmi,qpn}-age-sex-hc-aseg
# # with normative model
# ./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-decline-age-sex-case-aparc
# ./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
# ./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --min-age 55 --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
# ./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA_LATEST $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
# # simulated data
# ./scripts/split_train_test.py --n-splits 10 --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST {site1,site2,site3}-simulated

# combine for mega-analysis case
# no normative model (also no z-scoring)
parallel ./scripts/get_data-mega.py --tag decline-age-sex-case-hc-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni calgary pad ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni calgary pad ppmi qpn ::: {0..9}
# parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-55 --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# no normative model but with z-scoring
parallel ./scripts/get_data-mega.py --tag decline-age-sex-case-hc-aparc-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni calgary pad ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni calgary pad ppmi qpn ::: {0..9}
# # with normative model
# parallel ./scripts/get_data-mega.py --tag decline-age-sex-case-aparc-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-55-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc-aseg-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST adni ppmi qpn ::: {0..9}
# # simulated data
# parallel ./scripts/get_data-mega.py --tag simulated-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA_LATEST site1 site2 site3 ::: {0..9}

# create the nodes
fedbiomed component create -p ./fedbiomed/node-mega -c NODE
fedbiomed component create -p ./fedbiomed/node-adni -c NODE
fedbiomed component create -p ./fedbiomed/node-ppmi -c NODE
fedbiomed component create -p ./fedbiomed/node-qpn -c NODE
fedbiomed component create -p ./fedbiomed/node-pad -c NODE
fedbiomed component create -p ./fedbiomed/node-calgary -c NODE

# rename nodes to NODE-{site} in etc/config.ini (and optionally database filenames)

# add data to nodes
./scripts/add_data_to_nodes.py $DPATH_FL_DATA_LATEST $DPATH_FEDBIOMED

# start the nodes (in different terminal windows)
# in tmux: tmux new -s <dataset>; conda activate fl-pd
fedbiomed node -p ./fedbiomed/node-mega start
fedbiomed node -p ./fedbiomed/node-adni start
fedbiomed node -p ./fedbiomed/node-ppmi start
fedbiomed node -p ./fedbiomed/node-qpn start
fedbiomed node -p ./fedbiomed/node-pad start
fedbiomed node -p ./fedbiomed/node-calgary start

# run Fed-BioMed
./scripts/run_fedbiomed.py --tag decline-age-sex-case-hc-aparc-standardized --sgdc-loss log_loss --framework sklearn --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS $DPATH_FEDBIOMED $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-hc-aseg-standardized --framework sklearn --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS $DPATH_FEDBIOMED $FPATH_FEDBIOMED_CONFIG
# ./scripts/run_fedbiomed.py --tag simulated-standardized --dataset site1 --dataset site2 --dataset site3 --framework sklearn --n-splits 1 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG --sloppy --null

# run non-Fed-BioMed implementation
# no normative model (also no z-scoring)
./scripts/run_without_fedbiomed.py --tag decline-age-sex-case-hc-aparc  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# ./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-55 --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# ./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# no normative model but with z-scoring
./scripts/run_without_fedbiomed.py --tag decline-age-sex-case-hc-aparc-standardized  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-standardized --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# # with normative model
# ./scripts/run_without_fedbiomed.py --tag decline-age-sex-case-aparc-norm  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# # ./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-55-norm --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# ./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-norm --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS
# ./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg-norm  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA_LATEST $DPATH_FL_RESULTS

