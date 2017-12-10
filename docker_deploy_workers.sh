#!/bin/bash

echo "Which node do you want to use? [1, 2, 3, 4, 5]"
read node

for gpu in 0 1 2 3 4 5 6 7
do
    echo "GPU $gpu"
    # Build the image and label it on the Docker registry
    nvidia-docker build -t serrep$node.services.brown.edu:5000/contextual_circuit_bp .

    #Run the container
    nvidia-docker run -d --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs/cluster_projects/contextual_circuit_bp serrep$node.services.brown.edu:5000/contextual_circuit_bp bash start_gpu_worker.sh $gpu
done
