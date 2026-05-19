FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/app /input /output && chown -R user:user /opt/app /input /output && chmod -R 777 /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY requirements.txt /opt/app/
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user:user dataloaders/ /opt/app/dataloaders/
COPY --chown=user:user models/ /opt/app/models/
COPY --chown=user:user utils/ /opt/app/utils/
COPY --chown=user:user infer_wsi.py inference.sh /opt/app/
COPY --chown=user:user checkpoint/ /opt/app/checkpoint/

RUN chmod +x /opt/app/inference.sh

ENTRYPOINT ["/opt/app/inference.sh"]
