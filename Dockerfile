FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install -c bioconda --yes anarci scikit-learn pandas numpy matplotlib \
    && \
    conda clean -afy

RUN conda install -c bjrn --yes pandarallel 

RUN pip install stg

RUN wget https://www.dropbox.com/s/4sdu3b3u83voldy/finalized_model.sav

COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]

