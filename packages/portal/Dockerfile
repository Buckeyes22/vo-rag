FROM node:22-alpine@sha256:c610fcdfb1d5b4740dd70c284ed3cb16bb857e0f7166196e36a5501df7a3aa32 AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@10.33.0 --activate
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/portal/package.json packages/portal/
RUN pnpm install --frozen-lockfile --filter @vo-rag/portal
COPY packages/portal packages/portal
RUN pnpm --filter @vo-rag/portal build

FROM nginx:1.27-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10
COPY --from=build /app/packages/portal/dist /usr/share/nginx/html
COPY packages/portal/nginx.conf.template /etc/nginx/templates/default.conf.template
ENV VORAG_API_UPSTREAM=http://127.0.0.1:8001
EXPOSE 80
