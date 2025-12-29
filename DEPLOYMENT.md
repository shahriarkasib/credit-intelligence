# Credit Intelligence - Deployment Guide

This guide explains how to deploy the Credit Intelligence application with:
- **Frontend**: Vercel (Next.js)
- **Backend**: Railway (FastAPI + Python)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  VERCEL (Free)              │  RAILWAY (Free)               │
│  ─────────────────          │  ──────────────────────       │
│  Next.js Frontend           │  FastAPI Backend              │
│  - Real-time UI             │  - LangGraph Workflow         │
│  - WebSocket client         │  - WebSocket server           │
│  - Shows steps/logs         │  - Streams progress           │
└─────────────────────────────┴───────────────────────────────┘
```

## Prerequisites

1. GitHub account (for deployment)
2. Vercel account (free): https://vercel.com
3. Railway account (free): https://railway.app

## Step 1: Deploy Backend to Railway

### 1.1 Push to GitHub

First, push your code to GitHub:

```bash
cd /path/to/credit_intelligence
git add .
git commit -m "Add web deployment"
git push origin main
```

### 1.2 Deploy to Railway

1. Go to https://railway.app and sign in with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `credit_intelligence` repository
4. Railway will auto-detect the Python project

### 1.3 Configure Railway

In Railway dashboard:

1. Click on your service → Settings
2. Set **Root Directory**: `backend`
3. Set **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### 1.4 Add Environment Variables

In Railway Variables tab, add:

```
GROQ_API_KEY=your_groq_api_key
FINNHUB_API_KEY=your_finnhub_api_key
SEC_EDGAR_USER_AGENT=CreditIntelligence contact@example.com
MONGODB_URI=your_mongodb_connection_string
COURTLISTENER_API_KEY=your_courtlistener_key  # Optional
GOOGLE_SPREADSHEET_ID=your_spreadsheet_id  # Optional
```

### 1.5 Get Railway URL

After deployment, Railway gives you a URL like:
`https://credit-intelligence-production.up.railway.app`

Copy this URL for the frontend configuration.

## Step 2: Deploy Frontend to Vercel

### 2.1 Deploy to Vercel

1. Go to https://vercel.com and sign in with GitHub
2. Click "Add New Project"
3. Import your `credit_intelligence` repository
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`

### 2.2 Add Environment Variables

In Vercel project settings → Environment Variables:

```
NEXT_PUBLIC_API_URL=https://your-railway-app.up.railway.app
```

### 2.3 Deploy

Click "Deploy" and wait for the build to complete.

## Step 3: Test the Deployment

1. Open your Vercel URL (e.g., `https://credit-intelligence.vercel.app`)
2. Enter a company name (e.g., "Netflix")
3. Click "Run Analysis"
4. Watch the real-time progress in the UI!

## Local Development

### Run Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Run Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Troubleshooting

### CORS Errors
Make sure the backend CORS is configured correctly. The current config allows all origins (`*`). For production, update `backend/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app"],
    ...
)
```

### WebSocket Connection Failed
Check that:
1. Railway WebSocket is enabled (it is by default)
2. The `NEXT_PUBLIC_API_URL` is correct
3. No firewall blocking WebSocket connections

### Timeout Issues
Railway free tier has generous timeouts. If workflows still timeout:
1. Check Railway logs for errors
2. Consider upgrading to Railway Hobby plan ($5/mo) for more resources

## Costs

- **Vercel Hobby**: Free (10s function timeout, 100GB bandwidth)
- **Railway Starter**: Free ($5 credit/month, ~500 hours)
- **MongoDB Atlas**: Free (512MB storage)
- **Groq API**: Free (generous limits)

Total: **$0/month** for reasonable usage!

## Security Notes

1. Never commit `.env` files
2. Use environment variables for all secrets
3. For production, restrict CORS origins
4. Consider adding authentication (e.g., Clerk, NextAuth)
