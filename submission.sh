#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

readonly script_name="${0##*/}"

if [ "$#" -ne 1 ]; then
    echo "Usage: ${script_name} agent_name"
    exit 2;
fi

OUTPUT="jonas-drotleff_noah-wach_final-project"

mkdir -p $OUTPUT/${1}
find agent_code/${1} -maxdepth 1 -type f -exec cp -n {} ${OUTPUT}/${1} \;

# zip -r ${OUTPUT}.zip ${OUTPUT}

# rm -rf ${OUTPUT}

