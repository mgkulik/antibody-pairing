FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install -c bioconda --yes python=3.8 scikit-learn pandas numpy matplotlib 

RUN conda install -c conda-forge xgboost lightgbm \
    && \
    conda clean -afy

RUN conda install -c bjrn --yes pandarallel 

RUN pip install stg

RUN wget https://1drv.ms/u/s!AuXW2eqvluCBg5cOf0WyPfxpkbCjjA?e=IRhGqF

COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]

