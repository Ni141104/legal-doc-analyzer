# Legal Document Analyzer - Deployment Guide

This guide provides multiple deployment options for the Legal Document Analyzer application.

## 🚀 Quick Deployment Options

### Option 1: Vercel (Frontend) + Railway (Backend) - Recommended for MVP
**Pros**: Free tier available, easy setup, automatic HTTPS
**Cons**: Limited for enterprise scale

### Option 2: Google Cloud Platform - Enterprise Ready
**Pros**: Scalable, secure, full control
**Cons**: More complex setup, costs start immediately

### Option 3: Docker + VPS - Self-hosted
**Pros**: Full control, cost-effective for small teams
**Cons**: Requires DevOps knowledge

---

## 🎯 Option 1: Vercel + Railway (Recommended)

### Step 1: Deploy Backend to Railway

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `legal-doc-analyzer` repository

3. **Configure Backend Deployment**
   - Railway will auto-detect the backend
   - Set the following environment variables in Railway dashboard:
   ```
   PORT=8080
   CORS_ORIGINS=["https://your-vercel-app.vercel.app", "http://localhost:3000"]
   PYTHON_VERSION=3.11
   ```

4. **Custom Start Command**
   - In Railway settings, set start command:
   ```bash
   cd backend && python -m uvicorn simple_main:app --host 0.0.0.0 --port $PORT
   ```

### Step 2: Deploy Frontend to Vercel

1. **Create Vercel Account**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub

2. **Import Project**
   - Click "New Project"
   - Import your GitHub repository
   - Set root directory to `frontend`

3. **Configure Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
   ```

4. **Deploy**
   - Vercel will automatically build and deploy
   - Get your live URL

---

## 🏗️ Option 2: Google Cloud Platform

### Prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init
gcloud auth login
```

### Backend Deployment (Cloud Run)

1. **Create Dockerfile for Production**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   COPY backend/requirements.txt .
   RUN pip install -r requirements.txt

   # Copy backend code
   COPY backend/ .

   # Expose port
   EXPOSE 8080

   # Run the application
   CMD ["python", "-m", "uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

2. **Deploy to Cloud Run**
   ```bash
   cd backend
   gcloud run deploy legal-doc-analyzer \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="CORS_ORIGINS=https://your-frontend-domain.com"
   ```

### Frontend Deployment (Firebase Hosting)

1. **Install Firebase CLI**
   ```bash
   npm install -g firebase-tools
   firebase login
   ```

2. **Initialize Firebase**
   ```bash
   cd frontend
   firebase init hosting
   ```

3. **Build and Deploy**
   ```bash
   npm run build
   firebase deploy
   ```

---

## 🐳 Option 3: Docker Deployment

### 1. Create Production Dockerfiles

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/.next/static /usr/share/nginx/html
COPY --from=build /app/public /usr/share/nginx/html

EXPOSE 80
```

### 2. Docker Compose Setup

```yaml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - CORS_ORIGINS=["http://localhost:3000"]

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "3000:80"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8080
    depends_on:
      - backend
```

### 3. Deploy
```bash
docker-compose up -d
```

---

## 🔧 Environment Variables Reference

### Backend Environment Variables
```bash
# CORS Configuration
CORS_ORIGINS=["https://your-frontend-url.com"]

# Optional: Database (if you add persistence)
DATABASE_URL=postgresql://user:pass@host:port/db

# Optional: AI Services
OPENAI_API_KEY=your_openai_key
GOOGLE_AI_API_KEY=your_google_ai_key
```

### Frontend Environment Variables
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-backend-url.com

# Optional: Analytics
NEXT_PUBLIC_GA_ID=your_google_analytics_id
```

---

## 🚀 Quick Start Commands

### For Railway + Vercel:
1. Push your code to GitHub
2. Connect Railway to your repo (backend auto-deploys)
3. Connect Vercel to your repo (frontend auto-deploys)
4. Update CORS_ORIGINS in Railway with your Vercel URL
5. Update NEXT_PUBLIC_API_URL in Vercel with your Railway URL

### For Development:
```bash
# Backend
cd backend && python simple_main.py

# Frontend
cd frontend && npm run dev
```

---

## 📋 Deployment Checklist

- [ ] Choose deployment platform
- [ ] Set up backend deployment
- [ ] Configure environment variables
- [ ] Set up frontend deployment
- [ ] Update CORS configuration
- [ ] Test document upload functionality
- [ ] Test AI chat functionality
- [ ] Verify Full View and Download buttons work
- [ ] Set up custom domain (optional)
- [ ] Configure monitoring/logging (optional)

---

## 🆘 Troubleshooting

### Common Issues:

1. **CORS Errors**
   - Ensure CORS_ORIGINS includes your frontend URL
   - Check both http and https variants

2. **Build Failures**
   - Verify Node.js version (18+)
   - Check Python version (3.11+)
   - Ensure all dependencies are in requirements.txt/package.json

3. **API Connection Issues**
   - Verify NEXT_PUBLIC_API_URL points to correct backend
   - Check network policies on your hosting platform

4. **File Upload Issues**
   - Ensure backend has write permissions for temp files
   - Check file size limits on your hosting platform

---

## 🎯 Next Steps After Deployment

1. **Set up monitoring** (Error tracking, performance monitoring)
2. **Configure custom domain** (Optional)
3. **Set up CI/CD pipeline** (Automatic deployments on git push)
4. **Add authentication** (If needed for production)
5. **Set up database** (For persistent document storage)

Choose the deployment option that best fits your needs and budget!