
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source config/.env

# classification (cognitive decline)
./scripts/get_data-adni.py --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --decline --age --no-sex --no-diag --cases --no-controls --aparc --no-aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# classification (diagnosis)
./scripts/get_data-adni.py --no-decline --age --sex --diag --cases --controls --no-aparc --aseg  $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --diag --cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py --no-decline --age --sex --diag --cases --controls --no-aparc --aseg  $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA
#
./scripts/get_data-adni.py --no-decline --age --no-sex --diag --cases --controls --no-aparc --aseg  $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --no-sex --diag --cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py --no-decline --age --no-sex --diag --cases --controls --no-aparc --aseg  $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA
#
./scripts/get_data-adni.py --no-decline --age --sex --diag --cases --controls --aparc --no-aseg  $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --diag --cases --controls --aparc --no-aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py --no-decline --age --sex --diag --cases --controls --aparc --no-aseg  $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# regression (brain age)
./scripts/get_data-adni.py --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --no-decline --age --sex --no-diag --no-cases --controls --no-aparc --aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# simulated (MMSE)
./scripts/get_data-simulated.py $DPATH_FL_DATA

# split into training and testing sets
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-decline-age-case-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-diag-case-hc-aseg
./scripts/split_train_test.py --n-splits 10 --stratify-col DIAGNOSIS --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-diag-case-hc-aparc
./scripts/split_train_test.py --n-splits 10 --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {site1,site2,site3}-simulated

# combine for mega-analysis case
parallel ./scripts/get_data-mega.py --tag decline-age-case-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-diag-case-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-diag-case-hc-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag simulated --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA site1 site2 site3 ::: {0..9}

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
./scripts/run_fedbiomed.py --tag age-sex-diag-case-hc-aseg --sgdc-loss log_loss --framework sklearn --n-splits 10 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-diag-case-hc-aseg --sgdc-loss log_loss --framework sklearn --n-splits 10 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-hc-aseg --framework sklearn --n-splits 10 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag age-sex-diag-case-hc-aparc --framework sklearn --n-splits 1 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
./scripts/run_fedbiomed.py --tag simulated --dataset site1 --dataset site2 --dataset site3 --framework sklearn --n-splits 1 $DPATH_FL_DATA $DPATH_FL_RESULTS $DPATH_RESEARCHER $FPATH_FEDBIOMED_CONFIG
