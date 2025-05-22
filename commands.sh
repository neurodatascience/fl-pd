
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source config/.env

# classification (cognitive decline)
./scripts/get_data-adni.py --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# classification (diagnosis)
./scripts/get_data-adni.py --no-decline --age --sex --diag --cases --controls --aparc --aseg  $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --diag --cases --controls --aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py --no-decline --age --sex --diag --cases --controls --aparc --aseg  $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# regression (brain age)
./scripts/get_data-adni.py --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# simulated (MMSE)
./scripts/get_data-simulated.py $DPATH_FL_DATA

# split into training and testing sets
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --no-standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-decline-age-case-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --min-age 55 --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --no-standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc-aseg
./scripts/split_train_test.py --n-splits 10 --shuffle --standardize --random-state $RANDOM_SEED $DPATH_FL_DATA {site1,site2,site3}-simulated

# combine for mega-analysis case
parallel ./scripts/get_data-mega.py --tag decline-age-case-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg-55-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc-aseg-standardized --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
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
./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg-standardized  --n-rounds 1 --n-splits 1 --null 1 $DPATH_FL_DATA $DPATH_FL_RESULTS
./scripts/run_without_fedbiomed.py --tag age-sex-diag-case-hc-aparc-aseg  --n-rounds 1 --n-splits 1 --null 1 $DPATH_FL_DATA $DPATH_FL_RESULTS

