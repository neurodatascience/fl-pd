
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source .env

# classification (cognitive decline)
./scripts/get_data-adni.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --decline --age --no-sex --cases --no-controls --aparc --no-aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# regression (brain age)
./scripts/get_data-adni.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./scripts/get_data-ppmi.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./scripts/get_data-qpn.py  --no-decline --age --sex --no-cases --controls --no-aparc --aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA

# split into training and testing sets
./scripts/split_train_test.py --n-splits 10 --stratify-col COG_DECLINE --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-decline-age-case-aparc
./scripts/split_train_test.py --n-splits 10 --stratify-col AGE --shuffle --random-state $RANDOM_SEED $DPATH_FL_DATA {adni,ppmi,qpn}-age-sex-hc-aseg

# combine for mega-analysis case
parallel ./scripts/get_data-mega.py --tag decline-age-case-aparc --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}
parallel ./scripts/get_data-mega.py --tag age-sex-hc-aseg --suffix '-{}train' --random-state $RANDOM_SEED $DPATH_FL_DATA adni ppmi qpn ::: {0..9}

# create the nodes
fedbiomed component create -p ./fedbiomed/node-mega -c NODE
fedbiomed component create -p ./fedbiomed/node-adni -c NODE
fedbiomed component create -p ./fedbiomed/node-ppmi -c NODE
fedbiomed component create -p ./fedbiomed/node-qpn -c NODE

# add data to nodes
./scripts/add_data_to_nodes.py $DPATH_FL_DATA $DPATH_FEDBIOMED

# start the nodes (in different terminal windows)
fedbiomed node -p ./fedbiomed/node-mega start
fedbiomed node -p ./fedbiomed/node-adni start
fedbiomed node -p ./fedbiomed/node-ppmi start
fedbiomed node -p ./fedbiomed/node-qpn start
