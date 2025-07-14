
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source config/.env

# create single large TSV for dach dataset
./scripts/get_data-adni.py $FPATH_ADNI_PHENO $FPATH_ADNI_ASEG $FPATH_ADNI_APARC
./scripts/get_data-ppmi.py $FPATH_PPMI_PHENO $FPATH_PPMI_ASEG $FPATH_PPMI_APARC --fs7 --fpath-aseg-fs6 $FPATH_PPMI_ASEG_FS6 --fpath-aparc-fs6 $FPATH_PPMI_APARC_FS6
./scripts/get_data-qpn.py $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_ASEG $FPATH_QPN_APARC $DPATH_FL_DATA

# extract columns
./scripts/subset_data.py --dropna COG_DECLINE --decline --age --sex --no-diag --cases --no-controls --aparc --no-aseg $DPATH_FL_DATA {adni,ppmi,qpn}
./scripts/subset_data.py --dropna DIAGNOSIS --no-decline --age --sex --diag --cases --controls --aparc --aseg $DPATH_FL_DATA {adni,ppmi,qpn}
./scripts/subset_data.py --dropna AGE --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $DPATH_FL_DATA {adni,ppmi,qpn}

# also get the entire control subset
./scripts/subset_data.py --dropna AGE --dropna SEX --decline --age --sex --diag --no-cases --controls --aparc --aseg $DPATH_FL_DATA {adni,ppmi,qpn}

# simulated (MMSE)
./scripts/get_data-simulated.py $DPATH_FL_DATA

# split into training and testing sets
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-decline-age-sex-case-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --min-age 55 --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-decline-age-sex-case-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --min-age 55 --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --no-standardize --random-state $RANDOM_SEED --tag-adaptation decline-age-sex-diag-hc-aparc-aseg --no-norm $DPATH_FL_DATA $DPATH_NORMATIVE_MODELLING_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
./scripts/split_train_test.py --n-splits 10 --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {site1,site2,site3}-simulated

# combine for mega-analysis case
parallel ./scripts/get_data-mega.py --tag decline-age-sex-case-aparc-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-55-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc-aseg-norm --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag decline-age-sex-case-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-55 --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag simulated-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA site1 site2 site3 ::: {0..9}

# create the nodes
fedbiomed component create -p ./fedbiomed/node-mega -c NODE
fedbiomed component create -p ./fedbiomed/node-adni -c NODE
fedbiomed component create -p ./fedbiomed/node-ppmi -c NODE
fedbiomed component create -p ./fedbiomed/node-qpn -c NODE

# rename nodes to NODE-{site} (and optionally database filenames)

# add data to nodes
./scripts/add_data_to_nodes.py $DPATH_FL_DATA $DPATH_FEDBIOMED

# start the nodes (in different terminal windows)
fedbiomed node -p ./fedbiomed/node-mega start
fedbiomed node -p ./fedbiomed/node-adni start
fedbiomed node -p ./fedbiomed/node-ppmi start
fedbiomed node -p ./fedbiomed/node-qpn start

# run Fed-BioMed
./scripts/run_fedbiomed.py --tag decline-age-case-aparc --sgdc-loss log_loss --framework sklearn --n-splits 10 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-hc-aseg-standardized --framework sklearn --n-splits 1 --null 1 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-hc-aseg-55-standardized --framework sklearn --n-splits 1 --null 1 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg-standardized --sgdc-loss log_loss --framework sklearn --n-splits 1 --null 1 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag simulated-standardized --dataset site1 --dataset site2 --dataset site3 --framework sklearn --n-splits 1 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG --sloppy --null

# run non-Fed-BioMed implementation
./scripts/run_without_fedbiomed.py --tag decline-age-sex-case-aparc-norm  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-55-norm --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-norm --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg-norm  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag decline-age-sex-case-aparc  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg-55 --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-hc-aseg --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg  --n-rounds 3 --n-splits 10 --null 10 $DPATH_FL_DATA $DPATH_FL_RESULTS

