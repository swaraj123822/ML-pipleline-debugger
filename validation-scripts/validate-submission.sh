#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
    printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
    exit 1
fi

log() { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; exit 1; }

printf "\n${BOLD} OpenEnv Submission Validator${NC}\n\n"

# 1. PING CHECK
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30)

if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space is live and responds to /reset"
else
    fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200). Space might be asleep."
fi

# 2. DOCKER CHECK
log "${BOLD}Step 2/3: Running docker build${NC} ..."
if docker build "$REPO_DIR" > /dev/null 2>&1; then
    pass "Docker build succeeded"
else
    fail "Docker build failed"
fi

# 3. OPENENV CHECK
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
if openenv validate > /dev/null 2>&1; then
    pass "openenv validate passed"
else
    fail "openenv validate failed"
fi

printf "\n${GREEN}${BOLD} All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD} Your submission is ready to submit. ${NC}\n\n"
exit 0
