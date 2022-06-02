docker build -t human_attributes .
docker run \
--rm \
-it \
--gpus all \
-p 6006:6006 \
-v `pwd`/..:/workspace \
human_attributes \
bash 
