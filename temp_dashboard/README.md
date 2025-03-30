# MAEE Dashboard

This is the MAEE (Model Adversarial Example Evaluation) dashboard application.

## Deployment to Google Cloud Run

1. Build the Docker image:
```bash
gcloud builds submit --tag gcr.io/[PROJECT_ID]/maee-dashboard
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy maee-dashboard \
  --image gcr.io/[PROJECT_ID]/maee-dashboard \
  --platform managed \
  --region [REGION] \
  --allow-unauthenticated
```

Replace `[PROJECT_ID]` with your Google Cloud project ID and `[REGION]` with your desired region (e.g., `us-central1`).

## Local Development

To run the dashboard locally:

```bash
export MAEE_REPO_PATH=/path/to/demo-repo
streamlit run app.py
```

## Environment Variables

- `MAEE_REPO_PATH`: Path to the demo repository containing the MNIST model and data 