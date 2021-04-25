FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install -c bioconda --yes python=3.8 scikit-learn pandas numpy matplotlib 

RUN conda install -c conda-forge xgboost lightgbm \
    && \
    conda clean -afy

RUN conda install -c bjrn --yes pandarallel 

RUN pip install sgt

RUN wget https://www.dropbox.com/s/39w7jd7wxp2vujt/finalized_model.sav

COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]