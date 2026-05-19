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

COPY pyproject.toml /opt/app/
RUN pip install --user --no-cache-dir -e .

COPY --chown=user:user data/ /opt/app/data/
COPY --chown=user:user models/ /opt/app/models/
COPY --chown=user:user utils/ /opt/app/utils/
COPY --chown=user:user configs/ /opt/app/configs/
COPY --chown=user:user training/ /opt/app/training/
COPY --chown=user:user inference/ /opt/app/inference/
COPY --chown=user:user scripts/ /opt/app/scripts/
COPY --chown=user:user inference.sh /opt/app/
COPY --chown=user:user checkpoints/ /opt/app/checkpoints/

RUN chmod +x /opt/app/inference.sh

ENTRYPOINT ["/opt/app/inference.sh"]
