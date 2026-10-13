#!/usr/bin/env bash
# OCR worker pool for one RunPod pod (VOR-1434). Runs ON the pod in $WORKDIR.
#
#   run_units.sh run      OCR every unit in units/ (biggest first; resumable: ok units are skipped)
#   run_units.sh pack     write results.tar (ok JSONs + every note) for the coordinator to scp;
#                         prints PACKED and the status line
#   run_units.sh shrink   drop one worker from the live pool (GNU xargs SIGUSR2)
#   run_units.sh stop     stop the pool and every worker
#   run_units.sh status   one line: pool size, ok, failed, pending, done
#
# Lessons baked in (2026-09-26 incidents):
# - OOM kills: a worker averages 4.4 GB and peaks at 8.9 GB, and the limit that kills is the
#   container's cgroup (memory.max on v2, memory.limit_in_bytes on v1), not `free`. The pool
#   is sized at WORKER_PEAK_GB per worker from that limit; `shrink` lowers it live.
# - `pkill -f` matched and killed the ssh session running it: every pattern is bracketed.
# - tar streamed through ssh hung or was corrupted by stray stdout: `pack` writes a FILE.
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
VENV_PY=${VENV_PY:-$WORKDIR/venv/bin/python}
CGROUP_ROOT=${CGROUP_ROOT:-/sys/fs/cgroup}
MEMINFO=${MEMINFO:-/proc/meminfo}
WORKER_PEAK_GB=${WORKER_PEAK_GB:-9}
MAX_WORKERS=${MAX_WORKERS:-8}
# Same model cache bootstrap.sh warmed (and no stale token): workers never download.
HF_HOME=${HF_HOME:-$WORKDIR/hf}
export HF_HOME
unset HF_TOKEN
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# cgroup v1 reports "no limit" as a number near 2^63.
UNLIMITED_FLOOR=$((1 << 60))

memory_limit_bytes() {
    local v
    if [[ -r $CGROUP_ROOT/memory.max ]]; then
        v=$(<"$CGROUP_ROOT/memory.max")
        if [[ $v != max ]]; then echo "$v"; return; fi
    fi
    if [[ -r $CGROUP_ROOT/memory/memory.limit_in_bytes ]]; then
        v=$(<"$CGROUP_ROOT/memory/memory.limit_in_bytes")
        if ((v < UNLIMITED_FLOOR)); then echo "$v"; return; fi
    fi
    awk '/^MemTotal:/ { printf "%d\n", $2 * 1024 }' "$MEMINFO"
}

pool_size() {
    local limit per n
    limit=$(memory_limit_bytes)
    per=$((WORKER_PEAK_GB * 1024 * 1024 * 1024))
    n=$((limit / per))
    ((n > MAX_WORKERS)) && n=$MAX_WORKERS
    ((n < 1)) && n=1
    echo "$n"
}

unit_ok() {
    local base=$1
    [[ -s $base.json ]] && grep -q "ocr_worker: ok" "$base.note" 2>/dev/null
}

pending_units() {
    local f
    # ls -S: biggest file first, so the longest units start early.
    { ls -S units/*.pdf 2>/dev/null || true; } | while read -r f; do
        unit_ok "${f%.pdf}" || echo "$f"
    done
}

cmd_one() {
    local f=$1 base=${1%.pdf} name
    name=$(basename "$base")
    # The note appears only when the worker has ended (a running unit has none).
    "$VENV_PY" ocr_worker.py "$f" "$base.json" 2>>"$base.err" | tail -n 1 > "$base.note.tmp" || true
    mv -f "$base.note.tmp" "$base.note"
    if unit_ok "$base"; then echo "UNITOK $name" >> run.log; else echo "UNITFAIL $name" >> run.log; fi
    return 0
}

cmd_run() {
    local p
    p=$(pool_size)
    echo "$$" > run.pid
    echo "POOL $p limit=$(memory_limit_bytes)" >> run.log
    pending_units | xargs -r -P "$p" -I{} bash "$SELF" one {}
    echo ALLDONE >> run.log
    rm -f run.pid
}

cmd_pack() {
    local n listed=0
    : > pack.list
    for n in units/*.note; do
        [[ -e $n ]] || continue
        echo "$n" >> pack.list
        if unit_ok "${n%.note}"; then echo "${n%.note}.json" >> pack.list; listed=$((listed + 1)); fi
    done
    rm -f results.tar.part
    tar cf results.tar.part -T pack.list
    mv results.tar.part results.tar
    echo "PACKED ok=$listed"
    cmd_status
}

cmd_shrink() {
    # GNU xargs: SIGUSR2 lowers -P by one (never below 1).
    pkill -USR2 -f '[x]args -r -P' || true
}

cmd_stop() {
    local pid
    if [[ -s run.pid ]]; then
        pid=$(<run.pid)
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    pkill -TERM -f '[x]args -r -P' || true
    pkill -TERM -f '[o]cr_worker\.py' || true
    rm -f run.pid
}

cmd_status() {
    local ok=0 failed=0 pending=0 f alldone=no running=no
    for f in units/*.pdf; do
        [[ -e $f ]] || continue
        if unit_ok "${f%.pdf}"; then ok=$((ok + 1))
        elif [[ -e ${f%.pdf}.note ]]; then failed=$((failed + 1))
        else pending=$((pending + 1)); fi
    done
    grep -q '^ALLDONE' run.log 2>/dev/null && alldone=yes
    [[ -s run.pid ]] && kill -0 "$(<run.pid)" 2>/dev/null && running=yes
    echo "STATUS pool=$(pool_size) ok=$ok failed=$failed pending=$pending running=$running alldone=$alldone"
}

main() {
    cd "$WORKDIR"
    case "${1:-}" in
        run) cmd_run ;;
        one) cmd_one "$2" ;;
        pack) cmd_pack ;;
        shrink) cmd_shrink ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        *) echo "usage: run_units.sh run|pack|shrink|stop|status" >&2; return 2 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
