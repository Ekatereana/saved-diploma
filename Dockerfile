ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-alpine 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /

RUN apk update 
RUN apk add --no-cache --update \
    python3 gcc git python3-dev \
    gfortran musl-dev \
    libffi-dev openssl-dev

RUN pip install --upgrade pip

ENV PYTHONUNBUFFERED 1


RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8050

# Run the application.
CMD python3
