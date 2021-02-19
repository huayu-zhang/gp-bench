FROM continuumio/miniconda3

ARG TOOL_NAME=gp-bench
ARG ENV_NAME=graph_gp

COPY . /home/$TOOL_NAME

RUN conda env create -f /home/$TOOL_NAME/environment.yml
RUN echo "source activate "$ENV_NAME > ~/.bashrc
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

CMD "/bin/bash"
