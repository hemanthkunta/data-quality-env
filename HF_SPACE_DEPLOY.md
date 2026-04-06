# HF Space deploy runbook (Docker SDK)

## 1) Create Space
- Visibility: **Public**
- SDK: **Docker**
- Add tag: **openenv**

## 2) Push files
```bash
# ...existing code...
git add .
git commit -m "DataQualityEnv OpenEnv submission"
git push
```

## 3) Set Space secrets/variables
- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct`
- `HF_TOKEN=<your token>`
- `ENV_URL=http://localhost:7860`

## 4) Verify endpoints
```bash
curl https://<your-space>.hf.space/health
curl -X POST https://<your-space>.hf.space/reset \
  -H 'content-type: application/json' \
  -d '{"task_id":1,"seed":42}'
```

## 5) Validate submission
```bash
./validate-submission.sh https://<your-space>.hf.space
python scripts/check_graders.py  # run locally against local server first
```

## 6) Final checks
- `openenv validate` passes
- `/health` returns `{"status":"ok"}`
- `/reset` and `/step` both return valid JSON
- Inference completes under 20 minutes