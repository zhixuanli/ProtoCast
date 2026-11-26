#!/usr/bin/env bash
# Obtain the two upstream files used by the paper at their verified revision.
set -euo pipefail
cache="$(cd "$(dirname "$0")" && pwd)/.cache/predformer"
revision=a1493c4ade8eed740d7aed11da819e9aa7914c2d
url="https://raw.githubusercontent.com/yyyujintang/PredFormer/$revision/openstl"
mkdir -p "$cache"
stage="$(mktemp -d "$cache/download.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
curl --fail --silent --show-error --location "$url/modules/PredFormer_modules.py" -o "$stage/PredFormer_modules.py"
curl --fail --silent --show-error --location "$url/models/PredFormer_Quadruplet_TSST.py" -o "$stage/model.py"
# Load the sibling attention module and omit imports used only by upstream profiling.
sed -e '/^from openstl.utils import measure_throughput$/d' \
    -e '/^from fvcore.nn import FlopCountAnalysis, flop_count_table$/d' \
    -e 's/^from openstl.modules import /from PredFormer_modules import /' \
    "$stage/model.py" > "$stage/PredFormer_Quadruplet_TSST.py"
mv "$stage/PredFormer_modules.py" "$stage/PredFormer_Quadruplet_TSST.py" "$cache/"
