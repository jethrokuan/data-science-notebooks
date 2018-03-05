FROM jupyter/tensorflow-notebook

RUN conda install --quiet --yes \
    'pandas' \
    'gensim' \
    'snowballstemmer' \
    'nltk' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

Run conda install --quiet --yes \
    'feather-format' -c conda-forge && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Kaggle setup
RUN pip install kaggle

COPY kaggle.json /home/$NB_USER/.kaggle/kaggle.json
