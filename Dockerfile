COPY environment.yml /tmp/
RUN conda env update -q -f /tmp/environment.yml && \
    conda clean -y --all && \
    conda env export -n "root"