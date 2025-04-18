# Build stage
FROM python:3.10-alpine as builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    postgresql-dev \
    python3-dev \
    libffi-dev

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.10-alpine

# Create non-root user
RUN addgroup -S app && adduser -S app -G app

# Install runtime dependencies
RUN apk add --no-cache libpq

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 