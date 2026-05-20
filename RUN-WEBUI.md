cd /Users/max/Repos/PointsX
mkdir -p .dev-certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout .dev-certs/key.pem -out .dev-certs/cert.pem -days 365 \
  -subj "/CN=fitmeasure-local"

source .venv/bin/activate
pointsx-web --host 0.0.0.0 --port 8000 \
  --ssl-keyfile .dev-certs/key.pem --ssl-certfile .dev-certs/cert.pem