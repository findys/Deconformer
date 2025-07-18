FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
WORKDIR /workspace/
COPY deconformer/ /deconformer/
RUN pip3 install scanpy numpy pandas tqdm \
    && chmod a+x /deconformer/deconformer_predict.sh
ENTRYPOINT ["/bin/bash", "/deconformer/deconformer_predict.sh"]
CMD ["1"]