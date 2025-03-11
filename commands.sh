
# assumes working directory is the root of the project

# source the environment variables (paths)
set -a
source .env

# classification (cognitive decline)
./get_data-adni.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./get_data-ppmi.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
# ./get_data-ppmi.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg --fs6 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./get_data-qpn.py --decline --age --no-sex --cases --no-controls --aparc --no-aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA
./scripts/get_data-mega.py data adni ppmi qpn --tag decline-age-case-aparc

# regression (brain age)
./get_data-adni.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg $FPATH_ADNI_PHENO $FPATH_ADNI_IMAGING $DPATH_FL_DATA
./get_data-ppmi.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg --fs7 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
# ./get_data-ppmi.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg --fs6 --fpath-imaging-fs6 $FPATH_PPMI_IMAGING_FS6 $FPATH_PPMI_PHENO $FPATH_PPMI_IMAGING $DPATH_FL_DATA
./get_data-qpn.py --no-decline --age --sex --no-cases --controls --no-aparc --aseg $FPATH_QPN_DEMOGRAPHICS $FPATH_QPN_AGE $FPATH_QPN_DIAGNOSIS $FPATH_QPN_MOCA $FPATH_QPN_IMAGING $DPATH_FL_DATA
./scripts/get_data-mega.py data adni ppmi qpn --tag age-sex-hc-aseg
