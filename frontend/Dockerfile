# syntax=docker/dockerfile:1.7
# Web image: the React app built by Vite, served by unprivileged nginx, which also
# reverse-proxies /api, /health and /metrics to the API service (same origin => no CORS).
#   docker build -t multi-rag-web frontend
#   docker build --build-arg DIST=prebuilt -t multi-rag-web frontend   # use frontend/dist built on the host
ARG NODE_IMAGE=node:22-alpine
ARG NGINX_IMAGE=ghcr.io/nginxinc/nginx-unprivileged:1.27-alpine
ARG DIST=build

FROM ${NODE_IMAGE} AS build
WORKDIR /web
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY . .
# "." => API calls go to ./api/... on the same origin (nginx proxies them)
ARG VITE_API_BASE_URL=.
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM scratch AS prebuilt
COPY dist /web/dist

FROM ${DIST} AS dist

FROM ${NGINX_IMAGE}
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=dist /web/dist /usr/share/nginx/html
EXPOSE 8080
