FROM astrocrpublic.azurecr.io/runtime:3.3-2

USER root
RUN apt-get update && apt-get -y install git && apt-get clean
USER astro

# install your src package — pyproject.toml and src/ are already in build context
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .