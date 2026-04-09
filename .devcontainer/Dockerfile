# Official Rust image from Microsoft
FROM mcr.microsoft.com/devcontainers/rust:1

# Install Node.js and npm for frontend container commands in docker-compose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install WebAssembly tool (wasm-pack)
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh