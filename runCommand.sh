docker build -t watch-defect-app .
docker stop watch-defect-app
docker rm  watch-defect-app
docker run -d --gpus all --ipc=host -p 80:8080 --name watch-defect-app watch-defect-app --restart=always
docker logs -f --since=1m watch-defect-app