FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

ARG APP_DIR=/opt/app

ENV APP_DIR=${APP_DIR} \
    SYMBIOPAN_CKPT=${APP_DIR}/checkpoints/best_model.pth \
    SYMBIOPAN_SITE_CKPT=${APP_DIR}/checkpoints/site_classifier_atto.pth

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p ${APP_DIR} /input /output && chown -R user:user ${APP_DIR} /input /output && chmod -R 777 /output

USER user
WORKDIR ${APP_DIR}
ENV PATH="/home/user/.local/bin:${PATH}"

COPY pyproject.toml ${APP_DIR}/
RUN pip install --user --no-cache-dir -e .

COPY --chown=user:user symbiopan/ ${APP_DIR}/symbiopan/
COPY --chown=user:user configs/ ${APP_DIR}/configs/
COPY --chown=user:user scripts/ ${APP_DIR}/scripts/
COPY --chown=user:user inference.sh ${APP_DIR}/
COPY --chown=user:user checkpoints/ ${APP_DIR}/checkpoints/

RUN chmod +x ${APP_DIR}/inference.sh

ENTRYPOINT ["/opt/app/inference.sh"]
