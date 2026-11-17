#!/bin/bash
# Print a config value from config.local.yaml, falling back to config.yaml.
# Usage: bash src/af3tfm/config_get.sh <key>     (e.g. genome_fa, mail_user)
# Works for top-level keys and `paths:` entries written as one "key: value" line;
# inline "# comments" and surrounding quotes are stripped. Empty values count as unset.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "$DIR/config.local.yaml" "$DIR/config.yaml"; do
    [[ -f "$f" ]] || continue
    v=$(sed -n "s/^ *$1: *//p" "$f" | head -1 | sed -e 's/[[:space:]]#.*$//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")
    if [[ -n "$v" ]]; then echo "$v"; exit 0; fi
done
echo "config_get.sh: '$1' not set in $DIR/config{.local,}.yaml" >&2
exit 1
