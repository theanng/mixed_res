#!/usr/bin/env bash
# Phase Q tests (../phaseQ_relay_noise_analysis.md, logged in ../mixed_res_research_context.md Phase Q).
# Run from mixed_res_dit/. Each block skips run folders that already hold all 7 cases, so it can be re-run.
#
#   bash run_phaseQ.sh all        # x1 -> x34 -> x2 -> metrics, one GPU (~2.5-3 h)
#   bash run_phaseQ.sh x34        # X3 + X4: relabelling + where P2 is fixed     (26 runs, ~40-50 min)
#   bash run_phaseQ.sh x2         # X2: setup C with pixel-space operators       (27 runs, ~1.5 h, vae is slow)
#   bash run_phaseQ.sh x1         # X1: stock FLUX 512 vs 1024 texture period    (96 images, ~15-20 min)
#   bash run_phaseQ.sh metrics    # tables + phaseQ_out/metrics.csv (CPU)
#
# Two GPUs:   DEV=cuda:1 bash run_phaseQ.sh x1   alongside   bash run_phaseQ.sh x34 && bash run_phaseQ.sh x2
# Outputs:    phaseQ_out/x1/{512,1024}/*.png + scale_prior.json,  phaseQ_out/main/<run>/<case>/,  phaseQ_out/*.log
set -u
DEV=${DEV:-cuda:0}
OUT=${OUT:-./phaseQ_out}
MAIN="$OUT/main"
COMMON="--prompts comm --cats face_bisected periodic_bisected --scr_dump_streams --out_dir $MAIN"
E6="--native_ctx relay --scr_ctx crpa --scr_xscale taef1 --relay_gain 0"
mkdir -p "$MAIN"
FAILF=$(mktemp)          # failures, a file because the blocks run inside | tee subshells

mask() {
  case $1 in
    sb)       echo "--mask_pattern salblock  --strip_w 2 --mask_hr_ratio 0.1" ;;
    salcut)   echo "--mask_pattern salcut    --strip_w 2 --mask_hr_ratio 0.1" ;;
    salcheck) echo "--mask_pattern salcheck  --strip_w 2 --mask_hr_ratio 0.1" ;;
    saleyes)  echo "--mask_pattern saleyes   --strip_w 2 --mask_hr_ratio 0.03" ;;
    bs)       echo "--mask_pattern bisecting --mask_hr_ratio 0.3" ;;
    sc)       echo "--mask_pattern scattered --strip_w 2 --mask_hr_ratio 0.3" ;;
  esac
}

# R <tag_prefix> <er|noer> <mask> [run_seam flags...]   -> $MAIN/<tag_prefix><er|noer>/
R() {
  local tag=$1 only=$2 m=$3; shift 3
  local dir="$MAIN/${tag}${only}"
  local n; n=$(ls "$dir"/*/mixed.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge 7 ]; then echo "skip $dir ($n cases)"; return; fi
  echo "=== $(date +%H:%M:%S) ${tag}${only}"
  # shellcheck disable=SC2046
  python -m flux.run_seam --device "$DEV" $COMMON $(mask "$m") --only "$only" --tag_prefix "$tag" "$@" \
    || echo "${tag}${only}" >> "$FAILF"
}

x1() {
  echo "=== $(date +%H:%M:%S) X1 stock FLUX scale prior"
  python -m flux.scale_prior --device "$DEV" --out_dir "$OUT/x1" || echo "x1" >> "$FAILF"
}

x34() {
  # X3 -- is the relay's write relabelled as noise? Same flags except the relay gain; eps_H logged per step.
  R x3_sb_e6_    noer sb $E6 --scr_log
  R x3_sb_relay_ noer sb --native_ctx relay --scr_ctx crpa --scr_xscale taef1 --scr_band none --scr_log
  # X4 -- which half of E&R fixes P2 (ghost eyes)? A = plain CRPA; inner = LR shadows only; outer = HR band only.
  for m in sb bs salcut salcheck saleyes; do
    R "x4_${m}_A_"     noer "$m"
    R "x4_${m}_inner_" er   "$m" --pad 2 --pad_lr 2 --pad_hr 0
    R "x4_${m}_outer_" er   "$m" --pad 2 --pad_lr 0 --pad_hr 2
    R "x4_${m}_ER_"    er   "$m" --pad 2
    [ "$m" = sb ] || R "x4_${m}_e6_" noer "$m" $E6 --scr_log     # sb: x3_sb_e6
  done
}

x2() {
  # X2 -- setup C (single owner, HR band) with pixel-space proxies and an HR-owned final core.
  # band 2 first (the main comparison), then 1 and 4 (band width / blur contagion).
  for B in 2 1 4; do
    for m in sb bs sc; do
      for xs in learned taef1 vae; do
        local nx=""; [ "$xs" = learned ] || nx="--native_xscale $xs"
        # shellcheck disable=SC2086
        R "x2_${m}_${xs}_b${B}_" noer "$m" --native_ctx near --native_band_mode replace --native_band "$B" \
          $nx --native_final_core hr
      done
    done
  done
}

metrics() {
  python -m flux.scr_metrics "$MAIN" --csv "$OUT/metrics.csv" | tee "$OUT/metrics.txt"
  [ -f "$OUT/x1/scale_prior.json" ] && python -m flux.scale_prior --out_dir "$OUT/x1" --measure_only | tail -n 20
  return 0
}

case ${1:-all} in
  x1)      x1 2>&1 | tee -a "$OUT/x1.log" ;;
  x34)     x34 2>&1 | tee -a "$OUT/x34.log" ;;
  x2)      x2 2>&1 | tee -a "$OUT/x2.log" ;;
  metrics) metrics ;;
  all)     { x1; x34; x2; } 2>&1 | tee -a "$OUT/all.log"; metrics ;;
  *)       echo "usage: bash run_phaseQ.sh {all|x1|x34|x2|metrics}"; exit 1 ;;
esac
if [ -s "$FAILF" ]; then echo "FAILED: $(tr '\n' ' ' < "$FAILF")"; rm -f "$FAILF"; exit 1; fi
rm -f "$FAILF"
