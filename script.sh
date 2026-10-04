
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


hr
say "me ${EMAIL}"
curl -s  "${API}/api/v1/auth/me" -H "Authorization: Bearer ${TOKEN}"




# ================================================================
# UPLOAD + PROCESS
# ================================================================
hr
say "Upload test document"
TEST_FILE="./docs/SDR External NDA revised_waheeb_e.pdf"
# TEST_FILE="./docs/zerostrike_project (6).pdf"

DOC_TITLE="Python history ${RUN_ID}"

UPLOAD_RESPONSE=$(curl -s -X POST "${API}/api/v1/documents/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@${TEST_FILE}" \
  -F "title=${DOC_TITLE}")

echo "${UPLOAD_RESPONSE}" | pp
DOC_ID=$(printf '%s' "${UPLOAD_RESPONSE}" | jget document_id)
[ -n "${DOC_ID}" ] || die "upload did not return an id"
echo "    document_id=${DOC_ID}"


reprocess_response=$(curl -s -X POST "${API}/api/v1/documents/${DOC_ID}/reprocess" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"from_stage\":\"chunk\"}")


hr 
echo "reprocess_response = ${reprocess_response}"
hr
say "Polling document status (max 60s)"
STATUS=""
for i in $(seq .05 200); do
  DOC_JSON=$(curl -s "${API}/api/v1/documents/${DOC_ID}" \
    -H "Authorization: Bearer ${TOKEN}")
  STATUS=$(printf '%s' "${DOC_JSON}" | jget status)
  CHUNKS=$(printf '%s' "${DOC_JSON}" | jget chunk_count)
  current_stage=$(printf '%s' "${DOC_JSON}" | jget current_stage)
  echo "    [${i}] status=${STATUS} chunks=${CHUNKS} stage=${current_stage}"

  case "${STATUS}" in
    indexed) echo "    document processed"; break ;;
    failed)
      echo "    document FAILED:"
      echo "${DOC_JSON}" | pp
      die "document processing failed"
      ;;
  esac
  sleep 1

done


hr
say "Retrieve: in-document query with doc filter"
RETRIEVE_RESP=$(curl -s -X POST "${API}/api/v1/retrieval" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"How many core tables does the system design have?\",
    \"top_k\": 5,
    \"document_ids\": [\"${DOC_ID}\"]
  }")

echo "${RETRIEVE_RESP}" | pp

# Sanity: did we get passages?
PASSAGE_COUNT=$(printf '%s' "${RETRIEVE_RESP}" | python -c "
import sys, json
try:
    print(len(json.load(sys.stdin).get('passages', [])))
except Exception:
    print(0)
")
echo "    passages=${PASSAGE_COUNT}"
[ "${PASSAGE_COUNT}" -ge 1 ] || die "retrieval returned no passages"

# Top passage should mention the answer (nine core tables)
TOP_TEXT=$(printf '%s' "${RETRIEVE_RESP}" | python -c "
import sys, json
try:
    ps = json.load(sys.stdin).get('passages', [])
    print(ps[0]['text'].lower() if ps else '')
except Exception:
    print('')
")
if echo "${TOP_TEXT}" | grep -q "nine core tables\|9 core tables"; then
  echo "    ✓ top passage contains the expected answer"
else
  echo "    ✗ top passage does not mention 'nine core tables'"
  echo "    preview: $(printf '%s' "${TOP_TEXT}" | head -c 200)"
fi

# RRF score — not a similarity, just a rank signal. Top of a single
# ranking is 1/(60+1) ≈ 0.01639. Present in both lists doubles it.
TOP_SCORE=$(printf '%s' "${RETRIEVE_RESP}" | python -c "
import sys, json
try:
    ps = json.load(sys.stdin).get('passages', [])
    print(f'{ps[0][\"score\"]:.6f}' if ps else '')
except Exception:
    print('')
")
echo "    top_score=${TOP_SCORE} (RRF rank score, not cosine)"

CTX_LEN=$(printf '%s' "${RETRIEVE_RESP}" | python -c "
import sys, json
try:
    print(len(json.load(sys.stdin).get('context', '')))
except Exception:
    print(0)
")
echo "    context_chars=${CTX_LEN}"
[ "${CTX_LEN}" -gt 0 ] || die "retrieval returned empty context"

hr
say "Retrieve: no doc filter (searches all indexed docs)"
RETRIEVE_ALL=$(curl -s -X POST "${API}/api/v1/retrieval" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"query": "penetration testing framework", "top_k": 3}')

echo "${RETRIEVE_ALL}" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"    passages={len(d.get('passages', []))}\")
for p in d.get('passages', [])[:3]:
    print(f\"    [p.{p['page_start']}] {p['text'][:80].strip()}...\")
"

hr
say "Retrieve: query with no answer in corpus"
RETRIEVE_MISS=$(curl -s -X POST "${API}/api/v1/retrieval" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "top_k": 3}')

echo "${RETRIEVE_MISS}" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"    passages={len(d.get('passages', []))} (no relevance threshold by default)\")
"

hr
say "Retrieve: reranker status"
RETRIEVE_RERANK=$(curl -s -X POST "${API}/api/v1/retrieval" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many core tables does the system design have?", "top_k": 5}')

echo "${RETRIEVE_RERANK}" | python -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('reranked')
print(f\"    reranked={r}\")
if not r:
    print('    (no reranker configured — set RETRIEVAL__RERANK_URL and run TEI)')
print(f\"    timings_ms={d.get('timings_ms')}\")
"