#!/usr/bin/env bash
# Run a verification command offline with the old repositories/cache hidden.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v bwrap >/dev/null; then
    echo "This Linux isolation check requires bubblewrap (bwrap)." >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/validation/without_legacy_repos.sh COMMAND [ARGS...]" >&2
    exit 1
fi
options=(--bind / / --dev-bind /dev /dev --proc /proc --unshare-net)
for path in "$repo_root/../unitree_rl_lab-lambda" \
            "$repo_root/../unitree_mujoco-lambda" \
            "$repo_root/source" "${XDG_CACHE_HOME:-$HOME/.cache}/spark"; do
    if [[ -d "$path" ]]; then
        options+=(--tmpfs "$(realpath -- "$path")")
    fi
done
exec bwrap "${options[@]}" -- "$@"
