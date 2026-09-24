#!/bin/zsh
# THE 133 SCOREBOARD -- live, redrawing, every law with a verdict.
#
#   logs/tally.sh        LIVE. Redraws every 3 s until you press ctrl-C.
#   logs/tally.sh -1     print once and exit (for a pipe or a paste)
#   logs/tally.sh -u     live, but only the laws still UNFOUND
#
# It reads the run directory on disk, so it is stateless: start it, stop it and
# start it again whenever, and it always shows the truth as of now.
#
# THE VERDICT COLUMN:
#   FOUND     the fit crossed the stop bar and reported a model
#   UNFOUND   the fit ran its stage out and reported no answer
#   RUNNING   the fit for this law is on screen now
#   ...       not reached yet
#
# FOUND is the stop bar, NOT proof of the law -- read the model beside it.
set -u
ROOT=/Users/andrewmorgan/Dev/gamakon/fuller
ORDER=$ROOT/experiments/order133.txt
# THE LIVE RUN DIRECTORY, chosen by which one was written most recently — not by
# the order they are listed in. An older sweep's directory sits on disk beside
# the running one, and a picker that took the last name in a list reported a
# dead run's score while the live one was thirty laws further on.
C=$(ls -td $ROOT/logs/RUN $ROOT/logs/cascade133 2>/dev/null | while read -r d; do
      [ -d "$d/solved" ] && { echo "$d"; break; }
    done)
[ -n "$C" ] || { echo "no cascade on disk yet"; exit 0; }

GREEN=$'\033[32m'; RED=$'\033[31m'; BLUE=$'\033[34m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; OFF=$'\033[0m'

board() {
  local only_unfound=${1:-no}
  local total solved a60 a120 a180 cur curn alive pct
  total=$(grep -c . $ORDER)
  solved=$(ls $C/solved 2>/dev/null | wc -l | tr -d ' ')
  a60=$(ls $C/60s 2>/dev/null | wc -l | tr -d ' ')
  a120=$(ls $C/120s 2>/dev/null | wc -l | tr -d ' ')
  a180=$(ls $C/180s 2>/dev/null | wc -l | tr -d ' ')
  cur=$(ls -t $C/*/*/run.log 2>/dev/null | head -1)
  curn=$(echo $cur | awk -F/ '{print $(NF-1)}')
  if pgrep -f 'examples/evolve_fit' >/dev/null; then alive="${GREEN}RUNNING${OFF}"; else alive="${RED}STOPPED${OFF}"; fi
  pct=$(( solved * 100 / total ))

  printf '%s%s  THE 133%s   %b   %sFOUND%s %s%s%s / %s  (%s%%)   now: %s%s%s\n' \
    "$BOLD" "$(date +%H:%M:%S)" "$OFF" "$alive" \
    "$GREEN" "$OFF" "$BOLD" "$solved" "$OFF" "$total" "$pct" "$BLUE" "${curn:-none}" "$OFF"
  printf '%sattempted  60s %s   120s %s   180s %s   dir %s%s\n\n' \
    "$DIM" "$a60" "$a120" "$a180" "${C:t}" "$OFF"
  printf '%s%-9s %-26s %5s %7s  %s%s\n' "$DIM" VERDICT LAW STAGE GEN MODEL "$OFF"

  local n f s g m v col st
  while read -r n; do
    [ -n "$n" ] || continue
    f=""; s=""
    for st in 180s 120s 60s; do
      [ -f "$C/$st/$n/run.log" ] && { f=$C/$st/$n/run.log; s=$st; break; }
    done
    if [ -z "$f" ]; then
      [ "$only_unfound" = yes ] && continue
      printf '%s%-9s %-26s%s\n' "$DIM" "..." "$n" "$OFF"
      continue
    fi
    g=$(grep '^GENERATIONS' "$f" 2>/dev/null | awk '{print $2}')
    m=$(grep '^MODEL_INFIX' "$f" 2>/dev/null | cut -f2-)
    if [ -f "$C/solved/$n" ]; then v=FOUND; col=$GREEN
    elif [ -n "$g" ]; then v=UNFOUND; col=$RED
    else v=RUNNING; col=$BLUE; fi
    [ "$only_unfound" = yes ] && [ "$v" != UNFOUND ] && continue
    printf '%s%-9s%s %-26s %5s %7s  %.72s\n' "$col" "$v" "$OFF" "$n" "$s" "${g:--}" "${m:-—}"
  done < $ORDER
}

case "${1:-}" in
  -1) board ;;
  -u) while true; do printf '\033[H\033[2J'; board yes; sleep 3; done ;;
  *)  while true; do printf '\033[H\033[2J'; board; sleep 3; done ;;
esac
