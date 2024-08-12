docker build -t code-review-model .
docker run -p 8000:8000 -it code-review-model
curl -X POST "http://localhost:8000/review" -H "Content-Type: application/json" -d '{"code": "def hello_world():\n    print(\"Hello, world!\")"}'
