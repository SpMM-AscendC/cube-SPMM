#!/bin/bash

INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
OP_NAME=BcsrSpmmCustom
SOC_VERSION=Ascend910B2
OP_DIR=BcsrSpmmCustomOp

rm -rf ${OP_DIR}
${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i ${OP_NAME}.json -c ai_core-${SOC_VERSION} -lan cpp -out ./${OP_DIR}

cp -rf ${OP_NAME}/* ./${OP_DIR}
(cd ./${OP_DIR} && bash build.sh)
