FROM jupyter/tensorflow-notebook

ENV JUPYTER_PATH .:JUPYTER_PATH

RUN conda install --quiet --yes \
    'pandas' \
    'feather-format' \
    'gensim' \
    'snowballstemmer' \
    'nltk' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Kaggle setup
RUN pip install kaggle

COPY kaggle.json /home/$NB_USER/.kaggle/kaggle.json
