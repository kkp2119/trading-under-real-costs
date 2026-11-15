#!/bin/zsh
# Research 24 step 1: the collection chain (Kyle 2026-11-13: SPEC24_ANSWERS A1 and A2; SPEC24_QUESTIONS 14 approved by A8).
#   The development part and the sealed part are collected side by side, 3 connections each (N_p = 20, study24_big_gaps/universe24_<Y>.tsv).
#   Kyle (2026-11-13): it starts only after the universe D collection of research 23 has ended; the chain waits for it.
#   A2: the downloader stops when the free disk space falls below 25 GiB; a part is not started below 25 GiB either.
#   A part that stops for another reason (connection lost, more than 20 failed symbol-days) is run again after
#   5 minutes, at most 4 runs. A part that ends with failed symbol-days gets one more run. A stop by the free-disk
#   floor, by an existing file without a matching MANIFEST hash or by a leg list that differs is final.
#   Nothing is deleted or overwritten: a rerun skips every file whose sha256 equals its MANIFEST line.
#   No price, size or return value is printed: the downloader logs counts, seconds, bytes and hashes only.
#   (A copy of study21_more_pairs/bars1s/collect21.sh; changed: the two parts run side by side, the start condition, the wait.)
# Run from study24_big_gaps/bars1s:
#   nohup caffeinate -ims /bin/zsh collect24.sh > logs/collect24_nohup.out 2>&1 &
set -u
cd "$(dirname "$0")" || exit 9
PY=python
DATA="${DATA_DIR:-data}/research24_data"
NEED=25
mkdir -p logs
LOG=logs/collect24.log

say() { print -r -- "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$LOG"; }
free_gib() { df -k "$DATA" | awk 'NR==2 {printf "%.2f", $4/1048576}'; }

run_part() {   # split, connections
  local split=$1 conn=$2 try=1 extra=0 rc last free marker
  marker=logs/.collect24_${split}_marker
  while :; do
    free=$(free_gib)
    if (( free < NEED )); then
      say "$split: NOT STARTED: free disk $free GiB is below $NEED GiB (A2)"
      return 3
    fi
    say "$split: run $try starts, $conn connections, free disk $free GiB"
    touch "$marker"
    /bin/sleep 1
    "$PY" download.py --split "$split" --stocks --conn "$conn" --go >> "logs/collect24_${split}_stdout.txt" 2>&1
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
      if [[ -n "$last" ]] && grep -Eq "free disk space below|without a matching MANIFEST hash|differ from universe24|is outside every split" "$last"; then
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
n=0
while ps -axo command | grep -q "[d]ownload.py --split new --stocks"; do    # research 23's universe D collection
  (( n % 10 == 0 )) && say "waiting for the universe D collection of research 23 to end"
  n=$((n + 1))
  /bin/sleep 30
done
say "universe D collection is not running: starting dev and sealed side by side"
run_part dev 3 &
pid_dev=$!
/bin/sleep 5
run_part sealed 3 &
pid_sealed=$!
wait $pid_dev
rc_dev=$?
say "dev: finished with status $rc_dev"
wait $pid_sealed
rc_sealed=$?
say "sealed: finished with status $rc_sealed"
say "chain end: status dev $rc_dev, sealed $rc_sealed, free disk $(free_gib) GiB"
exit $(( rc_dev > rc_sealed ? rc_dev : rc_sealed ))
