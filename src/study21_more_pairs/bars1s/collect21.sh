#!/bin/zsh
# Research 21 step 1: the collection chain (Kyle 2026-11-11: SPEC21_ANSWERS A5 and A6, "개발 구간을 먼저 받아").
#   dev with 5 connections, then sealed with 3 connections (A6: at most 3 for the sealed part).
#   A5: before a part starts, the free disk space must be at least the estimate of what is still to be collected
#   + 12 GiB (estimate 11.7 GiB = dev 5.1 + sealed 6.6, results/estimate21_first_days.json): 23.7 GiB before dev,
#   18.6 GiB before sealed. The downloader itself stops below 9 GiB (SPEC21 1.3).
#   A part that stops for another reason (connection lost, more than 20 failed symbol-days) is run again after
#   5 minutes, at most 4 runs. A part that ends with failed symbol-days gets one more run. A stop by the free-disk
#   floor, by an existing file without a matching MANIFEST hash or by a leg list that differs is final.
#   Nothing is deleted or overwritten: a rerun skips every file whose sha256 equals its MANIFEST line.
#   No price, size or return value is printed: the downloader logs counts, seconds, bytes and hashes only.
# Run from study21_more_pairs/bars1s:
#   nohup caffeinate -ims /bin/zsh collect21.sh > logs/collect21_nohup.out 2>&1 &
set -u
cd "$(dirname "$0")" || exit 9
PY=python
DATA="${DATA_DIR:-data}/research21_data"
mkdir -p logs
LOG=logs/collect21.log

say() { print -r -- "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$LOG"; }
free_gib() { df -k "$DATA" | awk 'NR==2 {printf "%.2f", $4/1048576}'; }

run_part() {   # split, connections, GiB of free disk needed at the start
  local split=$1 conn=$2 need=$3 try=1 extra=0 rc last free marker
  marker=logs/.collect21_${split}_marker
  while :; do
    free=$(free_gib)
    if (( free < need )); then
      say "$split: NOT STARTED: free disk $free GiB is below the start condition $need GiB (A5)"
      return 3
    fi
    say "$split: run $try starts, $conn connections, free disk $free GiB (start condition $need GiB)"
    touch "$marker"
    /bin/sleep 1
    "$PY" download.py --split "$split" --stocks --conn "$conn" --go >> "logs/collect21_${split}_stdout.txt" 2>&1
    rc=$?
    last=$(find logs -name "download_${split}_stocks_*.log" -newer "$marker" | sort | tail -1)
    say "$split: run $try ended with exit code $rc, free disk $(free_gib) GiB; last line of ${last:-(no log)}: $(tail -1 "${last:-/dev/null}" | cut -c1-400)"
    if (( rc == 0 )); then
      if [[ -n "$last" ]] && tail -1 "$last" | grep -q 'failed 0 \[\]'; then
        say "$split: COMPLETE"
        return 0
      fi
      if (( extra )); then
        say "$split: ENDED with failed symbol-days after the extra run (errors.tsv)"
        return 0
      fi
      extra=1
      say "$split: failed symbol-days: one more run"
    else
      if [[ -n "$last" ]] && grep -Eq "free disk space below|without a matching MANIFEST hash|differ from universe21|is outside every split" "$last"; then
        say "$split: FINAL STOP (no rerun)"
        return 2
      fi
      if (( try >= 4 )); then
        say "$split: GAVE UP after $try runs"
        return 1
      fi
      /bin/sleep 300
    fi
    try=$((try + 1))
  done
}

say "chain start: pid $$, free disk $(free_gib) GiB"
run_part dev 5 23.7
rc=$?
say "dev: finished with status $rc"
if (( rc != 0 )); then
  say "chain end: sealed not started (dev status $rc)"
  exit $rc
fi
run_part sealed 3 18.6
rc=$?
say "sealed: finished with status $rc"
say "chain end: status $rc"
exit $rc
