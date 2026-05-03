#!/bin/bash
set -e
bash install.sh
./BcsrSpmmCustomOp/build_out/custom_opp_ubuntu_aarch64.run

(
cd AclNNInvocation
bash test_re_colcondense_balance_prof.sh
)