#!/usr/bin/env bash
# Create the GitHub repository and push this folder to it. Run from your Mac terminal:
#   cd ~/Desktop/canc/testis-spatial-shift && bash publish_to_github.sh
# The repo is created PRIVATE because the paper is under double-blind review; make it public after the decision
# (Settings -> Danger zone -> Change visibility, or: gh repo edit --visibility public).
set -euo pipefail
cd "$(dirname "$0")"
NAME="${1:-testis-spatial-shift}"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$NAME" --private --source=. --remote=origin --push \
    --description "Neighbourhood-aware cell typing in seminiferous tubules under tissue-organization and species shift (NeurIPS 2026 ml4spatialbio submission)"
  gh repo view --web
else
  cat <<EOF
GitHub CLI not found or not logged in. Two options:

  (a) install it and log in, then re-run this script:
        brew install gh && gh auth login

  (b) create the repository by hand:
        1. open https://github.com/new
        2. name: $NAME, visibility: Private, do NOT add a README/.gitignore/license (they already exist here)
        3. then run:
             git remote add origin https://github.com/<your-username>/$NAME.git
             git push -u origin main
EOF
fi
