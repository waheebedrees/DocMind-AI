#!/usr/bin/env bash
# script.sh — reset local state and run an end-to-end smoke test
#
# Usage:
#   ./script.sh                  # reset DB, Qdrant, Redis, then smoke test
#   SKIP_RESET=1 ./script.sh     # keep existing data, just run the smoke test

set -uo pipefail

# ---------- config ----------
API="${API:-http://localhost:8000}"
QDRANT="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${QDRANT_COLLECTION:-rag_collection}"
DB_NAME="${DB_NAME:-knowledge_db}"
DB_USER="${DB_USER:-postgres}"
SKIP_RESET="${SKIP_RESET:-0}"

EMAIL="me2@example.com"
PASSWORD="MyPass123"
USERNAME="me2"

UPLOAD_DIRS=("./uploads" "./data/uploads")
RUN_ID="$(date +%s)-$$"

# ---------- helpers ----------
hr()  { printf '\n%s\n' "------------------------------------------------------------"; }
say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

pp() {
  python -c 'import sys,json
try:
    print(json.dumps(json.load(sys.stdin), indent=2))
except Exception:
    sys.stdout.write("")
' || true
}

jget() {
  local key="$1"
  python -c "import sys,json
try:
    d = json.load(sys.stdin)
    v = d.get('${key}', '')
    print(v if v is not None else '')
except Exception:
    print('')
"
}



# ================================================================
# AUTH
# ================================================================
hr
say "Register ${EMAIL}"
REG_BODY=$(curl -s -w '\n%{http_code}' -X POST "${API}/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}")
REG_CODE=$(printf '%s' "${REG_BODY}" | tail -n1)
REG_JSON=$(printf '%s' "${REG_BODY}" | sed '$d')
echo "${REG_JSON}" | pp

case "${REG_CODE}" in
  200|201) echo "    registered" ;;
  409)     echo "    already registered, continuing" ;;
  *)       echo "    unexpected status ${REG_CODE}" ;;
esac

hr
say "Login ${EMAIL}"
LOGIN_RESPONSE=$(curl -s -X POST "${API}/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}")

echo "${LOGIN_RESPONSE}" | pp
TOKEN=$(printf '%s' "${LOGIN_RESPONSE}" | jget access_token)
[ -n "${TOKEN}" ] || die "could not obtain access_token"
echo "    token acquired (${#TOKEN} chars)"


# ================================================================
# UPLOAD + PROCESS
# ================================================================
hr
say "Upload test document"
TEST_FILE="/tmp/test-${RUN_ID}.txt"
DOC_TITLE="Python history ${RUN_ID}"
echo "Python was created by Guido van Rossum and first released in 1991. Run ${RUN_ID}." > "${TEST_FILE}"

UPLOAD_RESPONSE=$(curl -s -X POST "${API}/api/v1/documents/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@${TEST_FILE}" \
  -F "title=${DOC_TITLE}")

echo "${UPLOAD_RESPONSE}" | pp
DOC_ID=$(printf '%s' "${UPLOAD_RESPONSE}" | jget id)
[ -n "${DOC_ID}" ] || die "upload did not return an id"
echo "    document_id=${DOC_ID}"

hr
say "Polling document status (max 60s)"
STATUS=""
for i in $(seq 1 60); do
  DOC_JSON=$(curl -s "${API}/api/v1/documents/${DOC_ID}" \
    -H "Authorization: Bearer ${TOKEN}")
  STATUS=$(printf '%s' "${DOC_JSON}" | jget status)
  CHUNKS=$(printf '%s' "${DOC_JSON}" | jget chunk_count)
  echo "    [${i}] status=${STATUS} chunks=${CHUNKS}"

  case "${STATUS}" in
    completed) echo "    document processed"; break ;;
    failed)
      echo "    document FAILED:"
      echo "${DOC_JSON}" | pp
      die "document processing failed"
      ;;
  esac
  sleep 1
done
[ "${STATUS}" = "completed" ] || die "document did not reach 'completed' in 60s"


hr
say "Qdrant collection state"
curl -s "${QDRANT}/collections/${COLLECTION}" | pp


# ================================================================
# CHAT — smoke test
# ================================================================
hr
say "Chat: Who created Python?"
RESP1=$(curl -s -X POST "${API}/api/v1/chat/" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"message": "Who created Python?"}')

echo "${RESP1}" | pp
CONV_ID=$(printf '%s' "${RESP1}" | jget conversation_id)
echo "conversation_id=${CONV_ID}"

# Sanity check the score is a real cosine, not the old bug's 1.0
SCORE=$(printf '%s' "${RESP1}" | python -c "
import sys, json
d = json.load(sys.stdin)
src = d.get('message', {}).get('source', [])
print(src[0]['similarity_score'] if src else '')
")
if [ -n "${SCORE}" ]; then
  echo "    top similarity_score=${SCORE}"
  if [ "${SCORE}" = "1.0" ]; then
    echo "    WARN: score is exactly 1.0 — check RRF normalization"
  fi
fi

hr
say "Chat: When was it released? (follow-up, same conversation)"
RESP2=$(curl -s -X POST "${API}/api/v1/chat/" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"When was it released?\",\"conversation_id\":\"${CONV_ID}\"}")

echo "${RESP2}" | pp

hr
say "Smoke test complete"