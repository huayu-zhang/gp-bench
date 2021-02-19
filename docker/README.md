# To run the tools in a miniconda docker container

Clone the repo

```shell script
git clone git@git.ecdf.ed.ac.uk:hzhang13/graph_gp.git
cd graph_gp
```

Build the docker image, using the [Dockerfile](./docker/Dockerfile). Check the file for details on the image.

```shell script
docker build -t miniconda-gp_tools:1.0.0 -f ./docker/Dockerfile .
```

Or simply run

```shell script
bash docker_build.sh
```

After the build you can run the image in interactive mode -i with root -t. 

```shell script
docker run -it miniconda-gp_tools:1.0.0
```

You can now find the tools in */home/gp-bench* in the container with the graph_gp environment setup and activated. 

You can test or run the tools as the same as you do it locally. 
