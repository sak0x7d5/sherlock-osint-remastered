# Builds the source tree in this repository. Upstream's Dockerfile installed
# `sherlock-project` from PyPI, which would package their release rather than
# this derivative's code. See NOTICE.md.
#
# Release instructions:
  # 1. Update the VCS_REF tag to match the tagged version's FULL commit hash
  # 2. Build image with BOTH latest and version tags
    # i.e. `docker build -t sift-osint:0.16.0 -t sift-osint:latest .`
  #
  # VERSION_TAG is used for image labelling only; the installed version comes
  # from pyproject.toml, so the two cannot drift apart in the built artifact.

FROM python:3.13-slim-bookworm AS build
WORKDIR /src

RUN pip3 install --no-cache-dir --upgrade pip

COPY pyproject.toml LICENSE NOTICE.md ./
COPY docs/pyproject ./docs/pyproject
COPY sift_osint ./sift_osint

RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /dist .

FROM python:3.13-slim-bookworm
WORKDIR /sift

ARG VCS_REF= # CHANGE ME ON UPDATE
ARG VCS_URL="https://github.com/sak0x7d5/sift-osint"
ARG VERSION_TAG= # CHANGE ME ON UPDATE

ENV SIFT_ENV=docker

LABEL org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url=$VCS_URL \
      org.label-schema.name="sift-osint" \
      org.label-schema.version=$VERSION_TAG \
      org.label-schema.description="Independent derivative of the Sherlock Project"

COPY --from=build /dist/*.whl /tmp/
RUN pip3 install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# The stealth browser binary is not baked in; the first browser-backed run
# downloads it. Mount a persistent cache to avoid paying that per container.

ENTRYPOINT ["sift-osint"]
