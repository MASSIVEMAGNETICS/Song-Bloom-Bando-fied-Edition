# Enterprise Deployment Guide for SongBloom

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Production Readiness](#production-readiness)
- [Security & Compliance](#security--compliance)
- [Scalability & Performance](#scalability--performance)
- [Deployment Platforms](#deployment-platforms)
- [Monitoring & Operations](#monitoring--operations)
- [Disaster Recovery](#disaster-recovery)

## Overview

SongBloom Enterprise provides production-ready AI music generation with voice cloning capabilities. This guide covers enterprise deployment for web, mobile (iOS/Android), and API services.

### Key Features

✅ **Voice Cloning with Multiple Models**
- Dynamic model loading (on-device and server-based)
- Model registry with caching
- Quality validation and metrics

✅ **Enterprise-Grade Architecture**
- Microservices-based design
- Horizontal scalability
- Load balancing and auto-scaling
- High availability (99.9% uptime SLA)

✅ **Security & Compliance**
- End-to-end encryption
- SOC 2 Type II compliant architecture
- GDPR/CCPA ready
- Role-based access control (RBAC)

✅ **Multi-Platform Support**
- Web application (Streamlit/React)
- iOS mobile app
- Android mobile app
- RESTful API

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
│                     (AWS ALB / nginx)                        │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼───────┐   ┌───────▼───────┐
│  Web Frontend │   │   API Gateway   │   │ Mobile Apps   │
│  (Streamlit)  │   │   (FastAPI)     │   │  (iOS/And.)   │
└───────┬───────┘   └────────┬───────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                ┌────────────▼───────────┐
                │   Service Mesh         │
                │   (Backend Services)   │
                └────────────┬───────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼───────┐   ┌───────▼───────┐
│ Voice Cloning │   │ Music Generate │   │ Persona Mgmt  │
│   Service     │   │    Service     │   │   Service     │
└───────┬───────┘   └────────┬───────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                ┌────────────▼───────────┐
                │   Data Layer           │
                │   - PostgreSQL         │
                │   - Redis Cache        │
                │   - S3 Storage         │
                └────────────────────────┘
```

### Technology Stack

**Frontend:**
- Web: Streamlit / React.js
- Mobile: React Native / Flutter

**Backend:**
- API: FastAPI (Python 3.10+)
- Models: PyTorch 2.0+
- Cache: Redis
- Queue: Celery + RabbitMQ

**Infrastructure:**
- Container: Docker + Kubernetes
- Cloud: AWS / GCP / Azure
- CDN: CloudFlare
- Monitoring: Prometheus + Grafana

**Database:**
- Primary: PostgreSQL 14+
- Cache: Redis 7+
- Object Storage: S3 / MinIO

## Production Readiness

### Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| API Response Time | < 100ms | 95th percentile |
| Music Generation | < 45s | For 150s song |
| Voice Cloning | < 5s | Embedding extraction |
| Uptime | 99.9% | ~8.7 hours downtime/year |
| Concurrent Users | 10,000+ | Per cluster |

### Load Testing

```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz -L | tar xvz

# Run load test
k6 run scripts/load_test.js
```

Example load test script:

```javascript
// scripts/load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up
    { duration: '5m', target: 100 },   // Steady state
    { duration: '2m', target: 200 },   // Spike
    { duration: '5m', target: 200 },   // Steady state
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const payload = JSON.stringify({
    lyrics: 'Test lyrics for load testing',
    quality_preset: 'balanced',
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ${API_KEY}',
    },
  };

  const res = http.post('http://api.songbloom.ai/v1/generate', payload, params);
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });

  sleep(1);
}
```

### Health Checks

Implement comprehensive health checks:

```python
# app/health.py
from fastapi import APIRouter
from typing import Dict
import psutil
import torch

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict:
    """Basic health check"""
    return {"status": "healthy"}

@router.get("/health/detailed")
async def detailed_health_check() -> Dict:
    """Detailed system health"""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "database": await check_database(),
            "redis": await check_redis(),
            "model": await check_model(),
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
        }
    }

@router.get("/health/readiness")
async def readiness_check() -> Dict:
    """Kubernetes readiness probe"""
    # Check if all dependencies are ready
    if not await all_services_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@router.get("/health/liveness")
async def liveness_check() -> Dict:
    """Kubernetes liveness probe"""
    return {"status": "alive"}
```

## Security & Compliance

### Authentication & Authorization

```python
# app/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("30/minute")
async def generate_music(request: Request, user_id: str = Depends(verify_token)):
    # Implementation
    pass
```

### Data Encryption

```python
# app/encryption.py
from cryptography.fernet import Fernet
import base64
import os

class DataEncryption:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY') or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: bytes) -> str:
        """Encrypt data"""
        encrypted = self.cipher.encrypt(data)
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """Decrypt data"""
        decoded = base64.b64decode(encrypted_data.encode())
        return self.cipher.decrypt(decoded)

# Usage
encryption = DataEncryption()

# Encrypt voice embeddings before storage
encrypted_embedding = encryption.encrypt(voice_embedding.tobytes())

# Decrypt when needed
decrypted_embedding = encryption.decrypt(encrypted_embedding)
```

### Audit Logging

```python
# app/audit.py
import logging
from datetime import datetime
import json

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler('audit.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, user_id: str, action: str, resource: str, 
                  status: str, metadata: dict = None):
        """Log audit event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'status': status,
            'metadata': metadata or {}
        }
        self.logger.info(json.dumps(event))

# Usage
audit_logger = AuditLogger()
audit_logger.log_event(
    user_id="user123",
    action="CREATE",
    resource="voice_persona",
    status="SUCCESS",
    metadata={"persona_id": "abc123"}
)
```

## Scalability & Performance

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: songbloom-api
  namespace: songbloom-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: songbloom-api
  template:
    metadata:
      labels:
        app: songbloom-api
    spec:
      containers:
      - name: api
        image: songbloom/api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: songbloom-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: songbloom-api
  namespace: songbloom-prod
spec:
  selector:
    app: songbloom-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: songbloom-api-hpa
  namespace: songbloom-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: songbloom-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Caching Strategy

```python
# app/cache.py
import redis
import json
from functools import wraps
import hashlib

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def cache_result(ttl: int = 3600):
    """Cache decorator with TTL"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl=3600)
async def get_persona(persona_id: str):
    # Expensive database query
    return fetch_persona_from_db(persona_id)
```

## Monitoring & Operations

### Prometheus Metrics

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter('songbloom_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('songbloom_request_duration_seconds', 'Request duration')
active_generations = Gauge('songbloom_active_generations', 'Number of active music generations')
model_load_time = Histogram('songbloom_model_load_seconds', 'Model loading time')

# Middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    request_duration.observe(duration)
    
    return response
```

### Logging Configuration

```python
# app/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logging():
    """Configure production logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        'songbloom.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup PostgreSQL
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Backup Personas
aws s3 sync ./personas s3://songbloom-backups/personas/$DATE/

# Backup Models
aws s3 sync ./models s3://songbloom-backups/models/$DATE/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete

# Verify backup
gunzip -t "$BACKUP_DIR/db_$DATE.sql.gz"
if [ $? -eq 0 ]; then
    echo "Backup successful: $DATE"
else
    echo "Backup failed: $DATE" >&2
    exit 1
fi
```

### Recovery Procedures

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

# Restore database
gunzip -c "/backups/db_$BACKUP_DATE.sql.gz" | psql -h $DB_HOST -U $DB_USER $DB_NAME

# Restore personas
aws s3 sync s3://songbloom-backups/personas/$BACKUP_DATE/ ./personas/

# Restore models
aws s3 sync s3://songbloom-backups/models/$BACKUP_DATE/ ./models/

echo "Restore completed from backup: $BACKUP_DATE"
```

## Cost Optimization

### Resource Optimization

1. **GPU Utilization**
   - Use spot instances for non-critical workloads
   - Implement GPU sharing for multiple models
   - Auto-scale based on queue depth

2. **Storage Optimization**
   - Use lifecycle policies for S3
   - Compress audio files
   - Implement CDN for static assets

3. **Caching Strategy**
   - Cache frequent requests
   - Use CDN for media delivery
   - Implement edge caching

### Cost Monitoring

```python
# scripts/cost_monitor.py
import boto3
from datetime import datetime, timedelta

def get_monthly_cost():
    """Get current month AWS costs"""
    ce = boto3.client('ce')
    
    start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    response = ce.get_cost_and_usage(
        TimePeriod={'Start': start, 'End': end},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'SERVICE', 'Key': 'SERVICE'}]
    )
    
    return response['ResultsByTime']
```

## Support & Maintenance

### Support Tiers

| Tier | Response Time | Channels | SLA |
|------|---------------|----------|-----|
| Basic | 24 hours | Email | 95% uptime |
| Professional | 4 hours | Email, Chat | 99% uptime |
| Enterprise | 1 hour | Email, Chat, Phone | 99.9% uptime |

### Maintenance Windows

- **Scheduled Maintenance**: Every Sunday 2-4 AM UTC
- **Emergency Maintenance**: As needed with 2-hour notice
- **Rolling Updates**: Zero-downtime deployments

## Conclusion

This enterprise deployment guide provides a comprehensive framework for deploying SongBloom in production environments. Follow the security best practices, monitoring guidelines, and scalability patterns to ensure a robust, reliable service.

For additional support, contact: enterprise@songbloom.ai
