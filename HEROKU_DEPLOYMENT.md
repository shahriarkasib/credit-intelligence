# Heroku Deployment Guide

This guide explains how to deploy Credit Intelligence to Heroku.

## Prerequisites

- Heroku CLI installed
- Git repository initialized
- Heroku account

## Quick Deploy

### Option 1: Deploy via Heroku Button

[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

### Option 2: Manual Deploy via CLI

```bash
# Login to Heroku
heroku login

# Create a new app
heroku create your-app-name

# Set environment variables (see below)
heroku config:set GROQ_API_KEY=your-key

# Deploy
git push heroku main
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM inference | `gsk_xxxxx` |
| `MONGODB_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.mongodb.net/dbname` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | - |
| `LANGSMITH_PROJECT` | LangSmith project name | `credit-intelligence` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_HOST` | Langfuse host URL | `https://cloud.langfuse.com` |
| `TAVILY_API_KEY` | Tavily API key for web search | - |
| `PARALLEL_API_KEY` | Parallel AI API key | - |

### Google Sheets Logging (Optional)

For Google Sheets logging on Heroku, you need to encode your service account credentials as base64:

1. Get your Google service account JSON file
2. Encode it as base64:
   ```bash
   base64 -i your-credentials.json | tr -d '\n'
   ```
3. Set the environment variable:
   ```bash
   heroku config:set GOOGLE_SHEETS_CREDENTIALS="<base64-encoded-json>"
   heroku config:set GOOGLE_SPREADSHEET_ID="your-spreadsheet-id"
   ```

| Variable | Description |
|----------|-------------|
| `GOOGLE_SHEETS_CREDENTIALS` | Base64-encoded Google service account JSON |
| `GOOGLE_SPREADSHEET_ID` | Google Sheets spreadsheet ID |

## Setting Environment Variables

### Via Heroku CLI

```bash
# Set single variable
heroku config:set GROQ_API_KEY=your-key

# Set multiple variables
heroku config:set \
  GROQ_API_KEY=your-key \
  MONGODB_URI=your-mongodb-uri \
  LANGSMITH_API_KEY=your-langsmith-key
```

### Via Heroku Dashboard

1. Go to your app's Settings tab
2. Click "Reveal Config Vars"
3. Add each variable

## Files Required for Deployment

The following files are required for Heroku deployment:

- `Procfile` - Defines the web process
- `runtime.txt` - Specifies Python version (3.11.9)
- `requirements.txt` - Python dependencies
- `app.json` - App metadata and config vars

## Architecture Notes

### Backend API

The backend runs as a FastAPI application using uvicorn:
```
web: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

### Frontend

The frontend is a Next.js application. For full deployment, you may need to:
1. Deploy the frontend separately (e.g., Vercel)
2. Or build and serve it from the backend

### WebSocket Support

WebSocket connections are supported for real-time updates during credit assessments. Heroku's router supports WebSockets.

## Scaling

```bash
# Scale dynos
heroku ps:scale web=1

# Check dyno status
heroku ps
```

## Logs

```bash
# View logs
heroku logs --tail

# View specific process logs
heroku logs --tail --dyno web
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure PYTHONPATH is set correctly in app.json
2. **Google Sheets not working**: Verify base64 encoding of credentials
3. **MongoDB connection failed**: Check MONGODB_URI format and whitelist Heroku IPs

### Health Check

The API provides a health endpoint:
```bash
curl https://your-app.herokuapp.com/health
```

Expected response:
```json
{"status": "healthy", "active_runs": 0, "active_connections": 0}
```
