# ==========================================
# STAGE 1: WebAssembly Engine (Rust)
# ==========================================
FROM rust:1.77-slim AS wasm-builder
WORKDIR /app/wasm-engine
# Install dependencies for wasm-pack
RUN apt-get update && apt-get install -y curl pkg-config libssl-dev
RUN cargo install wasm-pack
COPY ./wasm-engine .
RUN wasm-pack build --target web --release

# ==========================================
# STAGE 2: Frontend (React)
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY ./frontend/package*.json ./
RUN npm install
COPY ./frontend .
# Copy the WASM package generated from Stage 1 into the frontend
COPY --from=wasm-builder /app/wasm-engine/pkg ./src/wasm-engine/pkg
RUN npm run build

# ==========================================
# STAGE 3: Production (Lightweight NGINX Server)
# ==========================================
FROM nginx:alpine
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]